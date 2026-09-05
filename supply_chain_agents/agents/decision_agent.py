"""
Decision / Recommendation agent — Section 4.3, expanded to the fuller spec:
multi-objective optimization, full ranking (not just a single winner),
business-rule awareness, confidence scoring, risk-level display, and
alternative plans for when a human rejects the top recommendation.

Input:  DisruptionEvent + list[CandidateOutcome]
Output: ActionRecommendation

Design tip from the plan: this is the one agent where LLM reasoning genuinely
adds value — use it to weigh trade-offs and phrase the justification, but the
underlying numbers always come from the simulation, never invented by the model.

REQUIRES these schema additions (not included in this file — ask for
schemas.py separately):
  - Severity extended to Literal["low", "medium", "high", "critical"]
    (to match the expanded monitoring agent)
  - ActionRecommendation gets new fields:
      confidence: float = 0.8
      risk_level: Literal["low", "medium", "high"] = "medium"
      ranking: list[str] = []                        # action labels, best -> worst
      business_rules_satisfied: bool = True
      excluded_by_rules: dict[str, list[str]] = {}    # action -> violated rule names
Until schemas.py has these, this file will raise validation errors on
construction — expected, and resolves once that update lands.

What's new here vs. the previous version:
  - Multi-objective optimization: the objective isn't hardcoded to "minimize
    cost" anymore — `DecisionConfig.objective_priority` can be "cost",
    "delay", or "service_level", so priorities can flip (e.g. cost-first
    today, speed-first during a peak season) without touching logic.
  - Hard business rules, separate from the soft severity-based risk
    ceiling: a budget cap, a minimum service level, and an absolute
    (never-exceed-regardless-of-severity) stockout-risk cap. Rules that
    disqualify a candidate outright are tracked per-candidate for
    explainability, not just silently dropped.
  - Full ranking, not just a single winner — every rule-compliant candidate
    is ordered 1st/2nd/3rd/..., which is what makes "if rejected, show the
    next best" possible without re-running the simulation.
  - Confidence score: how clearly the winner beat the runner-up on the
    chosen objective, discounted when the winner only qualifies as the
    least-bad option rather than a genuinely good one.
  - Risk-level classification (low/medium/high) for the dashboard, derived
    from the chosen candidate's stockout risk.
  - `build_recommendation_card()`: a small convenience function matching the
    plan's "Recommendation Card" example (action, expected savings, risk
    reduction, delay) — optional, for whichever module ends up rendering it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import uuid

from schemas import DisruptionEvent, CandidateOutcome, ActionRecommendation, Severity
from llm_client import call_llm

_ID_NAMESPACE = uuid.UUID("6f6f6f66-6f66-6f66-6f66-6f6f6f6f6f6f")

ObjectivePriority = Literal["cost", "profit", "delay", "service_level"]


@dataclass
class BusinessRules:
    """Hard constraints — Section: 'Never violate: safety stock, supplier
    constraints, budget.' Unlike the soft severity-based risk ceiling below,
    a candidate that fails one of these is disqualified outright, not just
    deprioritized."""
    max_stockout_risk: float = 0.95      # absolute ceiling, regardless of disruption severity
    min_service_level: float = 0.40      # absolute floor
    max_cost_delta: float | None = None  # optional budget cap; None = no cap


@dataclass
class DecisionConfig:
    """Configurable objective (Section 4.3) — both *what* counts as
    risk-feasible (soft, severity-aware) and *what* to optimize for once
    feasible (the objective priority) are tunable without touching logic."""
    base_stockout_risk_ceiling: float = 0.5
    severity_risk_ceiling: dict[Severity, float] = field(
        default_factory=lambda: {"critical": 0.15, "high": 0.30, "medium": 0.50, "low": 0.70}
    )
    objective_priority: ObjectivePriority = "profit"
    business_rules: BusinessRules = field(default_factory=BusinessRules)
    risk_level_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 0.15, "medium": 0.40}  # >= medium threshold => "high"
    )

    def ceiling_for(self, severity: Severity) -> float:
        return self.severity_risk_ceiling.get(severity, self.base_stockout_risk_ceiling)


DEFAULT_CONFIG = DecisionConfig()


# ---------------------------------------------------------------------------
# Business rules
# ---------------------------------------------------------------------------

def _rule_violations(candidate: CandidateOutcome, rules: BusinessRules) -> list[str]:
    """Returns the names of every hard rule this candidate violates (empty
    list = fully compliant). Tracking *which* rule failed, not just a bool,
    is what makes the recommendation explainable when a candidate gets
    disqualified."""
    violations = []
    if candidate.stockout_risk > rules.max_stockout_risk:
        violations.append(f"stockout_risk > {rules.max_stockout_risk:.0%}")
    if candidate.service_level_impact < rules.min_service_level:
        violations.append(f"service_level < {rules.min_service_level:.0%}")
    if rules.max_cost_delta is not None and candidate.cost_delta > rules.max_cost_delta:
        violations.append(f"cost > budget cap ₹{rules.max_cost_delta:,.0f}")
    return violations


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _objective_value(candidate: CandidateOutcome, priority: ObjectivePriority) -> float:
    """Lower is always better in this function's output, regardless of
    priority, so a single sort direction works for every objective.

    NOTE: "cost" (cost_delta) only counts direct operational spend (rush
    fees, holding costs) -- it does NOT include revenue lost to a worse
    service level. "profit" (profit_impact = -cost_delta - lost_sales_effect)
    is the fuller picture and should usually be preferred as the default;
    optimizing on "cost" alone can pick an action that's cheaper to execute
    but net-worse for the business once lost sales are counted.
    """
    if priority == "delay":
        return candidate.delivery_delay_days
    if priority == "service_level":
        return -candidate.service_level_impact  # maximize service level = minimize its negation
    if priority == "profit":
        return -candidate.profit_impact  # maximize profit_impact = minimize its negation
    return candidate.cost_delta  # default: "cost"


def rank_candidates(
    candidates: list[CandidateOutcome],
    severity: Severity,
    config: DecisionConfig = DEFAULT_CONFIG,
) -> tuple[list[CandidateOutcome], bool, bool, dict[str, list[str]]]:
    """
    Returns (ranking, constraint_satisfied, business_rules_satisfied, excluded_by_rules).

    ranking:                   every business-rule-compliant candidate, best -> worst.
                                Risk-feasible candidates (within the severity-based
                                ceiling) are always ranked ahead of non-feasible ones;
                                within each group, ordered by the configured objective.
    constraint_satisfied:      True if the #1-ranked candidate meets the soft ceiling.
    business_rules_satisfied:  True if at least one candidate is fully rule-compliant.
    excluded_by_rules:         action label -> violated hard-rule names, for every
                                candidate that failed at least one hard rule.
    """
    excluded_by_rules: dict[str, list[str]] = {}
    compliant: list[CandidateOutcome] = []
    for c in candidates:
        violations = _rule_violations(c, config.business_rules)
        if violations:
            excluded_by_rules[c.action] = violations
        else:
            compliant.append(c)

    business_rules_satisfied = bool(compliant)
    # Never crash and never return nothing: if every candidate violates a hard
    # rule, fall back to ranking all of them anyway — flagged clearly via
    # business_rules_satisfied=False — because a human still needs *something*
    # to review, even if it's "here's the least-bad option we have."
    pool = compliant if compliant else list(candidates)

    ceiling = config.ceiling_for(severity)
    ranking = sorted(
        pool,
        key=lambda c: (
            0 if c.stockout_risk <= ceiling else 1,          # risk-feasible bucket first
            _objective_value(c, config.objective_priority),   # then by chosen objective
            c.cost_delta,                                     # tiebreaker
        ),
    )
    constraint_satisfied = bool(ranking) and ranking[0].stockout_risk <= ceiling
    return ranking, constraint_satisfied, business_rules_satisfied, excluded_by_rules


def _risk_level(stockout_risk: float, config: DecisionConfig) -> str:
    if stockout_risk < config.risk_level_thresholds["low"]:
        return "low"
    if stockout_risk < config.risk_level_thresholds["medium"]:
        return "medium"
    return "high"


def _confidence(
    ranking: list[CandidateOutcome],
    priority: ObjectivePriority,
    constraint_satisfied: bool,
    business_rules_satisfied: bool,
) -> float:
    """How clearly the #1 choice beats the runner-up on the chosen
    objective. Discounted when the winner is only a forced/least-bad
    choice — a confident-sounding recommendation would be misleading there."""
    if len(ranking) < 2:
        confidence = 0.95  # no competing option to be uncertain against
    else:
        best = _objective_value(ranking[0], priority)
        runner_up = _objective_value(ranking[1], priority)
        gap = abs(runner_up - best) / (abs(runner_up) + 1e-6)
        confidence = min(0.99, 0.55 + gap)

    if not business_rules_satisfied:
        confidence = min(confidence, 0.40)
    elif not constraint_satisfied:
        confidence = min(confidence, 0.60)
    return round(confidence, 2)


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def _make_action_id(event_id: str, chosen_action: str) -> str:
    raw = f"{event_id}:{chosen_action}"
    return f"ACT-{uuid.uuid5(_ID_NAMESPACE, raw).hex[:8]}"


def _fallback_justification(
    ranking: list[CandidateOutcome],
    constraint_satisfied: bool,
    business_rules_satisfied: bool,
    risk_level: str,
    objective_priority: ObjectivePriority = "cost",
) -> str:
    chosen = ranking[0]
    others = ranking[1:]

    if not business_rules_satisfied:
        lead = "No candidate satisfies the configured business rules — showing the least-bad option. "
    elif not constraint_satisfied:
        lead = "No candidate met the risk ceiling — this is the least-bad option. "
    else:
        lead = ""

    # Savings must be expressed in whichever metric actually drove the
    # ranking -- cost_delta and profit_impact routinely disagree (a cheaper
    # cost_delta can still be the worse profit_impact once lost sales are
    # counted), so hardcoding one here would contradict the chosen action.
    if others:
        if objective_priority == "profit":
            next_best = max(c.profit_impact for c in others)  # higher profit_impact = better
            savings = round(chosen.profit_impact - next_best, 2)
            savings_clause = f"Saves ~₹{max(savings, 0):,.0f} in profit impact vs. next-best alternative, "
        else:
            next_best_cost = min(c.cost_delta for c in others)
            savings = round(next_best_cost - chosen.cost_delta, 2)
            savings_clause = f"Saves ~₹{max(savings, 0):,.0f} vs. next-best alternative, "
    else:
        savings_clause = "Only compliant candidate available, "

    return (
        f"{lead}{savings_clause}keeps stockout risk at {chosen.stockout_risk * 100:.0f}% "
        f"({risk_level} risk), adds {chosen.delivery_delay_days} day(s) delivery delay, "
        f"service level {chosen.service_level_impact * 100:.0f}%."
    )


def build_recommendation_card(rec: ActionRecommendation) -> dict:
    """Convenience structure matching the plan's 'Recommendation Card'
    example (action / expected savings / risk reduction / delay). Not used
    by the orchestrator — for whichever module ends up rendering this."""
    chosen = next(c for c in rec.all_candidates if c.action == rec.chosen_action)
    others = [c for c in rec.all_candidates if c.action != rec.chosen_action]
    if others:
        best_alt_cost = min(c.cost_delta for c in others)
        best_alt_risk = min(c.stockout_risk for c in others)
        savings = max(0.0, best_alt_cost - chosen.cost_delta)
        risk_reduction_pct = max(0.0, (best_alt_risk - chosen.stockout_risk)) * 100
    else:
        savings, risk_reduction_pct = 0.0, 0.0
    return {
        "recommended_action": rec.chosen_action,
        "expected_savings": round(savings, 2),
        "risk_reduction_pct": round(risk_reduction_pct, 1),
        "delay_days": chosen.delivery_delay_days,
        "confidence": rec.confidence,
        "risk_level": rec.risk_level,
    }


def get_alternative(rec: ActionRecommendation, already_tried: set[str] | None = None) -> str | None:
    """'If recommendation rejected -> return second best / third best.'
    Since the full ranking is stored on the recommendation, no re-simulation
    is needed — just walk to the next action not yet tried."""
    already_tried = already_tried or {rec.chosen_action}
    for action in rec.ranking:
        if action not in already_tried:
            return action
    return None  # every ranked option has already been tried/rejected


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    event: DisruptionEvent,
    candidates: list[CandidateOutcome],
    config: DecisionConfig = DEFAULT_CONFIG,
) -> ActionRecommendation:
    ranking, constraint_satisfied, business_rules_satisfied, excluded_by_rules = rank_candidates(
        candidates, event.severity, config
    )
    chosen = ranking[0]
    risk_level = _risk_level(chosen.stockout_risk, config)
    confidence = _confidence(ranking, config.objective_priority, constraint_satisfied, business_rules_satisfied)
    fallback = _fallback_justification(
        ranking, constraint_satisfied, business_rules_satisfied, risk_level, config.objective_priority
    )

    candidate_summary = "\n".join(
        f"- {c.action}: cost ₹{c.cost_delta:,.0f}, delay {c.delivery_delay_days}d, "
        f"stockout risk {c.stockout_risk * 100:.0f}%, service level {c.service_level_impact * 100:.0f}%"
        for c in candidates
    )
    ceiling = config.ceiling_for(event.severity)
    status_note = (
        f"risk ceiling {ceiling * 100:.0f}%" if constraint_satisfied
        else f"NOTE: no candidate met the {ceiling * 100:.0f}% risk ceiling — mention this trade-off"
    )
    
    # --- MULTI-AGENT DEBATE ENGINE ---
    # 1. Logistics Agent (Focuses on Stockout Risk and Service Level)
    logistics_prompt = (
        f"You are the Logistics Manager. A disruption occurred: {event.type} on {event.affected_id}.\n"
        f"Candidates:\n{candidate_summary}\n\n"
        f"Argue in ONE sentence for the candidate with the lowest stockout risk and best service level, regardless of cost."
    )
    logistics_arg = call_llm(logistics_prompt, fallback="I recommend prioritizing the safest option to ensure customer satisfaction.")

    # 2. Finance Agent (Focuses on Cost)
    finance_prompt = (
        f"You are the Finance Manager. A disruption occurred: {event.type} on {event.affected_id}.\n"
        f"Candidates:\n{candidate_summary}\n\n"
        f"Argue in ONE sentence for the candidate with the lowest cost, regardless of stockout risk."
    )
    finance_arg = call_llm(finance_prompt, fallback="I recommend the cheapest option to preserve our profit margins.")

    # 3. Chief Orchestrator (Synthesizes and makes final call)
    orchestrator_prompt = (
        f"You are the Chief Supply Chain Officer. A disruption occurred: {event.type} on {event.affected_id} "
        f"(severity {event.severity}, objective: {config.objective_priority}, {status_note})\n"
        f"Candidates:\n{candidate_summary}\n\n"
        f"The system algorithm has chosen: {chosen.action}.\n\n"
        f"Logistics Manager argues: {logistics_arg}\n"
        f"Finance Manager argues: {finance_arg}\n\n"
        f"Synthesize this debate and explain in ONE sentence why you agree that the algorithm's chosen action ({chosen.action}) is the best trade-off."
    )
    orchestrator_arg = call_llm(orchestrator_prompt, fallback=fallback)
    
    # Format the final justification with HTML for the frontend modal
    justification = (
        f"<strong>Logistics Agent:</strong> {logistics_arg}<br><br>"
        f"<strong>Finance Agent:</strong> {finance_arg}<br><br>"
        f"<strong>Chief Orchestrator:</strong> {orchestrator_arg}"
    )

    return ActionRecommendation(
        action_id=_make_action_id(event.event_id, chosen.action),
        event_id=event.event_id,
        chosen_action=chosen.action,
        candidates_considered=len(candidates),
        justification=justification,
        status="pending_approval",
        all_candidates=candidates,
        constraint_satisfied=constraint_satisfied,
        business_rules_satisfied=business_rules_satisfied,
        confidence=confidence,
        risk_level=risk_level,
        ranking=[c.action for c in ranking],
        excluded_by_rules=excluded_by_rules,
    )