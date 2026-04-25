"""Boundary tests for the Pydantic v2 contracts in fishtrade.models.

Each model_validator must be exercised by at least one positive and one
negative test. The contract is the line between the system's modules —
if validation here is loose, every downstream agent inherits the bug.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fishtrade.models import (
    DebateResult,
    DebateTurn,
    ExecutionResult,
    FillInfo,
    HardCheckResult,
    IndicatorScore,
    NavSnapshot,
    Order,
    PortfolioSnapshot,
    Position,
    ResearchReport,
    RiskDecision,
    SoftJudgment,
    VarResult,
)
from tests.conftest import make_indicator, make_indicator_list


# --------------------------------------------------------------------------- #
# IndicatorScore
# --------------------------------------------------------------------------- #


class TestIndicatorScore:
    def test_minimal_valid(self):
        ind = make_indicator()
        assert ind.score == 1
        assert ind.is_degraded is False

    def test_degraded_must_be_score_zero(self):
        with pytest.raises(ValidationError, match="is_degraded=True 时 score 必须为 0"):
            IndicatorScore(
                name="PE_RATIO",
                display_name_zh="市盈率",
                display_name_en="PE Ratio",
                raw_value=None,
                score=1,
                reasoning="missing data",
                is_degraded=True,
                degrade_reason="yfinance returned None",
            )

    def test_degraded_requires_reason(self):
        with pytest.raises(ValidationError, match="degrade_reason"):
            IndicatorScore(
                name="PE_RATIO",
                display_name_zh="市盈率",
                display_name_en="PE Ratio",
                raw_value=None,
                score=0,
                reasoning="missing data",
                is_degraded=True,
                # degrade_reason omitted
            )

    @pytest.mark.parametrize("bad_score", [-2, 2, 5, 100])
    def test_score_must_be_in_minus1_0_1(self, bad_score):
        with pytest.raises(ValidationError):
            make_indicator(score=bad_score)

    def test_reasoning_max_length_enforced(self):
        with pytest.raises(ValidationError):
            IndicatorScore(
                name="PE_RATIO",
                display_name_zh="市盈率",
                display_name_en="PE Ratio",
                raw_value=10.0,
                score=0,
                reasoning="x" * 401,
            )

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            IndicatorScore(
                name="PE_RATIO",
                display_name_zh="市盈率",
                display_name_en="PE Ratio",
                raw_value=10.0,
                score=0,
                reasoning="ok",
                made_up_field="oops",  # type: ignore[call-arg]
            )


# --------------------------------------------------------------------------- #
# ResearchReport — the crown jewel: total_score, verdict, and degrade gating
# --------------------------------------------------------------------------- #


def _build_report(
    *,
    scores: list[int] | None = None,
    total_score: int | None = None,
    verdict: str = "BUY",
    confidence: float = 0.7,
    is_facet_degraded: bool = False,
    n_highlights: int = 3,
) -> ResearchReport:
    if scores is None:
        scores = [1] * 5 + [0] * 5  # sums to 5 -> BUY
    if total_score is None:
        total_score = sum(scores)
    return ResearchReport(
        facet="fundamental",
        ticker="AAPL",
        as_of_date="2026-04-25",
        indicator_scores=make_indicator_list(scores),
        total_score=total_score,
        verdict=verdict,  # type: ignore[arg-type]
        confidence=confidence,
        key_highlights=[f"highlight {i}" for i in range(n_highlights)],
        industry_class="growth",
        is_facet_degraded=is_facet_degraded,
        degrade_summary="degraded" if is_facet_degraded else None,
    )


class TestResearchReport:
    def test_happy_path_buy(self):
        r = _build_report()
        assert r.total_score == 5
        assert r.verdict == "BUY"

    @pytest.mark.parametrize(
        "scores,expected_verdict",
        [
            ([1] * 5 + [0] * 5, "BUY"),       # 5 -> BUY
            ([1] * 6 + [0] * 4, "BUY"),       # 6 -> BUY
            ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], "HOLD"),   # 1 -> HOLD
            ([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], "HOLD"),   # 4 -> HOLD
            ([0] * 10, "SELL"),                # 0 -> SELL
            ([-1] * 5 + [0] * 5, "SELL"),      # -5 -> SELL
        ],
    )
    def test_verdict_thresholds(self, scores, expected_verdict):
        r = _build_report(scores=scores, verdict=expected_verdict)
        assert r.verdict == expected_verdict
        assert r.total_score == sum(scores)

    def test_total_score_mismatch_rejected(self):
        with pytest.raises(ValidationError, match="total_score"):
            _build_report(scores=[1] * 5 + [0] * 5, total_score=7)

    @pytest.mark.parametrize(
        "scores,wrong_verdict",
        [
            ([1] * 5 + [0] * 5, "HOLD"),   # actual BUY
            ([1] * 5 + [0] * 5, "SELL"),
            ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], "BUY"),  # actual HOLD
            ([0] * 10, "BUY"),                         # actual SELL
        ],
    )
    def test_verdict_mismatch_rejected(self, scores, wrong_verdict):
        with pytest.raises(ValidationError, match="verdict"):
            _build_report(scores=scores, verdict=wrong_verdict)

    def test_indicator_count_must_be_exactly_10(self):
        with pytest.raises(ValidationError):
            ResearchReport(
                facet="fundamental",
                ticker="AAPL",
                as_of_date="2026-04-25",
                indicator_scores=make_indicator_list([1, 1, 1]),
                total_score=3,
                verdict="HOLD",
                confidence=0.5,
                key_highlights=["a", "b", "c"],
            )

    def test_facet_degraded_caps_confidence(self):
        with pytest.raises(ValidationError, match="confidence"):
            _build_report(is_facet_degraded=True, confidence=0.9)

    def test_facet_degraded_low_confidence_ok(self):
        r = _build_report(is_facet_degraded=True, confidence=0.4)
        assert r.is_facet_degraded is True

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            _build_report(confidence=1.5)
        with pytest.raises(ValidationError):
            _build_report(confidence=-0.1)

    @pytest.mark.parametrize("n", [2, 6])
    def test_highlights_length_bounds(self, n):
        with pytest.raises(ValidationError):
            _build_report(n_highlights=n)


# --------------------------------------------------------------------------- #
# Debate
# --------------------------------------------------------------------------- #


def _bull_turn(round_: int = 0, conclusion: str = "BUY") -> DebateTurn:
    return DebateTurn(
        round=round_,
        role="bull",
        argument="bullish argument",
        cited_indicators=["PE_RATIO", "MACD"],
        conclusion=conclusion,  # type: ignore[arg-type]
    )


class TestDebateTurn:
    def test_minimal_valid(self):
        t = _bull_turn()
        assert t.role == "bull"

    def test_cited_indicators_must_not_be_empty(self):
        with pytest.raises(ValidationError):
            DebateTurn(
                round=0,
                role="bull",
                argument="hi",
                cited_indicators=[],
                conclusion="BUY",
            )

    def test_cited_indicators_must_be_non_empty_strings(self):
        with pytest.raises(ValidationError, match="空字符串"):
            DebateTurn(
                round=0,
                role="bull",
                argument="hi",
                cited_indicators=["PE_RATIO", "  "],
                conclusion="BUY",
            )

    @pytest.mark.parametrize("bad_round", [-1, 4, 99])
    def test_round_bounds(self, bad_round):
        with pytest.raises(ValidationError):
            _bull_turn(round_=bad_round)

    def test_argument_max_length(self):
        with pytest.raises(ValidationError):
            DebateTurn(
                round=0,
                role="bull",
                argument="x" * 2001,
                cited_indicators=["PE_RATIO"],
                conclusion="BUY",
            )


class TestDebateResult:
    def _result(
        self, verdict: str = "BUY", pct: float = 5.0
    ) -> DebateResult:
        return DebateResult(
            turns=[_bull_turn()],
            final_verdict=verdict,  # type: ignore[arg-type]
            final_rationale="rationale",
            confidence=0.6,
            proposed_position_pct=pct,
        )

    def test_buy_with_position(self):
        r = self._result("BUY", 5.0)
        assert r.proposed_position_pct == 5.0

    def test_buy_zero_position_rejected(self):
        with pytest.raises(ValidationError, match="proposed_position_pct"):
            self._result("BUY", 0.0)

    @pytest.mark.parametrize("verdict", ["HOLD", "SELL"])
    def test_hold_sell_must_be_zero_position(self, verdict):
        with pytest.raises(ValidationError, match="必须为 0"):
            self._result(verdict, 5.0)

    def test_position_pct_upper_bound(self):
        with pytest.raises(ValidationError):
            self._result("BUY", 10.5)


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #


class TestRiskSchemas:
    def _hard(self, rule="R1_POSITION_LIMIT", passed=True):
        return HardCheckResult(rule=rule, passed=passed, actual=5.0, threshold=10.0, detail="ok")

    def _var(self, passed=True, sample_size=120):
        return VarResult(
            var_95=0.034,
            portfolio_impact=0.0034,
            passed=passed,
            sample_size=sample_size,
        )

    def _soft(self, adjustment="keep", pct=5.0):
        return SoftJudgment(
            flags=["NONE"],
            adjustment=adjustment,
            adjusted_position_pct=pct,
            reasoning="ok",
        )

    def test_hard_check_minimal(self):
        h = self._hard()
        assert h.passed is True

    def test_hard_check_unknown_rule_rejected(self):
        with pytest.raises(ValidationError):
            HardCheckResult(rule="R9_FAKE", passed=True, detail="x")  # type: ignore[arg-type]

    def test_var_zero_sample_implies_failed(self):
        with pytest.raises(ValidationError, match="sample_size=0"):
            VarResult(var_95=0, portfolio_impact=0, passed=True, sample_size=0)

    def test_var_zero_sample_failure_ok(self):
        v = VarResult(
            var_95=0,
            portfolio_impact=0,
            passed=False,
            sample_size=0,
            fallback_reason="insufficient",
        )
        assert v.passed is False

    def test_soft_flags_required(self):
        with pytest.raises(ValidationError):
            SoftJudgment(
                flags=[],
                adjustment="keep",
                adjusted_position_pct=5.0,
                reasoning="ok",
            )

    def test_soft_reject_must_be_zero_position(self):
        with pytest.raises(ValidationError, match="reject"):
            self._soft(adjustment="reject", pct=5.0)

    def test_risk_decision_approve(self):
        rd = RiskDecision(
            decision="approve",
            adjusted_position_pct=5.0,
            hard_checks=[self._hard()],
            var_result=self._var(),
            soft_judgment=self._soft(),
        )
        assert rd.decision == "approve"

    def test_risk_decision_reject_requires_reason(self):
        with pytest.raises(ValidationError, match="reject_reason"):
            RiskDecision(
                decision="reject",
                adjusted_position_pct=0.0,
                hard_checks=[self._hard(passed=False)],
                var_result=self._var(),
                soft_judgment=self._soft(adjustment="reject", pct=0.0),
            )

    def test_risk_decision_reject_must_have_zero_position(self):
        with pytest.raises(ValidationError, match="adjusted_position_pct"):
            RiskDecision(
                decision="reject",
                adjusted_position_pct=5.0,
                hard_checks=[self._hard(passed=False)],
                var_result=self._var(),
                soft_judgment=self._soft(adjustment="reject", pct=0.0),
                reject_reason="hard rule failed",
            )

    def test_risk_decision_hard_checks_required_non_empty(self):
        with pytest.raises(ValidationError):
            RiskDecision(
                decision="approve",
                adjusted_position_pct=5.0,
                hard_checks=[],
                var_result=self._var(),
                soft_judgment=self._soft(),
            )


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


class TestExecutionSchemas:
    def _buy(self, **overrides) -> Order:
        kwargs = dict(
            symbol="AAPL",
            side="buy",
            qty=10,
            limit_price=200.0,
            stop_price=190.0,
        )
        kwargs.update(overrides)
        return Order(**kwargs)  # type: ignore[arg-type]

    def test_order_buy_with_stop(self):
        o = self._buy()
        assert o.stop_price == 190.0

    def test_order_sell_must_not_have_stop(self):
        with pytest.raises(ValidationError, match="sell"):
            Order(symbol="AAPL", side="sell", qty=10, limit_price=200.0, stop_price=190.0)

    def test_order_qty_must_be_positive(self):
        with pytest.raises(ValidationError):
            self._buy(qty=0)

    def test_order_price_must_be_positive(self):
        with pytest.raises(ValidationError):
            self._buy(limit_price=0.0)

    def test_execution_failed_requires_error(self):
        with pytest.raises(ValidationError, match="failed"):
            ExecutionResult(mode="paper", order=self._buy(), status="failed")

    def test_execution_filled_requires_fill_info(self):
        with pytest.raises(ValidationError, match="fill_info"):
            ExecutionResult(mode="paper", order=self._buy(), status="filled")

    def test_execution_skipped_must_not_have_order(self):
        with pytest.raises(ValidationError, match="skipped"):
            ExecutionResult(mode="dryrun", order=self._buy(), status="skipped")

    def test_execution_skipped_no_order_ok(self):
        e = ExecutionResult(mode="dryrun", order=None, status="skipped")
        assert e.status == "skipped"

    def test_execution_filled_ok(self):
        e = ExecutionResult(
            mode="paper",
            order=self._buy(),
            status="filled",
            fill_info=FillInfo(avg_price=199.5, filled_qty=10, fill_time="2026-04-25T10:00:00Z"),
        )
        assert e.fill_info is not None


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #


class TestPortfolioSchemas:
    def test_minimal_snapshot(self):
        p = PortfolioSnapshot(cash=100_000.0, nav=100_000.0)
        assert p.positions == []

    def test_snapshot_with_positions(self):
        p = PortfolioSnapshot(
            cash=80_000.0,
            positions=[Position(symbol="AAPL", qty=100, avg_cost=200.0)],
            nav=100_000.0,
            nav_history=[NavSnapshot(date="2026-04-24", nav=100_000.0)],
        )
        assert len(p.positions) == 1

    def test_snapshot_rejects_duplicate_symbols(self):
        with pytest.raises(ValidationError, match="重复 symbol"):
            PortfolioSnapshot(
                cash=0.0,
                positions=[
                    Position(symbol="AAPL", qty=10, avg_cost=100.0),
                    Position(symbol="AAPL", qty=20, avg_cost=110.0),
                ],
                nav=100_000.0,
            )

    def test_max_drawdown_bounds(self):
        with pytest.raises(ValidationError):
            PortfolioSnapshot(cash=0.0, nav=0.0, max_drawdown_pct=-1.0)
        with pytest.raises(ValidationError):
            PortfolioSnapshot(cash=0.0, nav=0.0, max_drawdown_pct=200.0)

    def test_position_qty_positive(self):
        with pytest.raises(ValidationError):
            Position(symbol="AAPL", qty=0, avg_cost=100.0)

    def test_nav_snapshot_negative_rejected(self):
        with pytest.raises(ValidationError):
            NavSnapshot(date="2026-04-25", nav=-1.0)


# --------------------------------------------------------------------------- #
# GraphState — TypedDict shape only (no validators), but we sanity-check it
# --------------------------------------------------------------------------- #


def test_graph_state_imports_and_supports_partial_writes():
    from fishtrade.models.state import GraphState

    state: GraphState = {
        "run_id": "abc",
        "warnings": [],
        "tokens_total": 0,
    }
    assert state["run_id"] == "abc"
    state["warnings"].append("X")
    assert state["warnings"] == ["X"]
