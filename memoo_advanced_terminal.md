# Memoo Advanced Terminal — Technical Specification

**Version:** 1.0  
**Date:** 2026-02-28  
**Status:** Final Draft  
**Target Platform:** Solana Blockchain (Meme Tokens)

---

## 1. Product Overview

Memoo Advanced Terminal is a high-frequency trading terminal for Meme tokens on Solana. It provides real-time, low-latency state updates for thousands of active tokens, delivering actionable signals based on **momentum analysis**, **behavioral detection**, and **stability control**.

**Core Value Proposition:**

| Pillar | Description |
|---|---|
| **Early Discovery** | Capture token lifecycle signals at ignition phase |
| **Stable Confirmation** | Multi-validation filtering of false signals |
| **Fast Risk Avoidance** | Sub-second crash warnings |

---

## 2. Global Architecture & Rules

All backend logic **must** adhere to these foundational principles:

| Rule | Specification |
|---|---|
| **Base Currency** | All financial values (price, market cap, liquidity, volume) normalized to **USD** |
| **Global Tick** | Full recalculation & push every **5 seconds** |
| **Rolling Windows** | All time-based metrics use continuous sliding windows (not fixed snapshots) |
| **Anti-Flicker** | *"1 hit = noise, 2 hits = signal"* — all state transitions require 2 consecutive ticks (10s), except panic crash |
| **Precision** | All percentages and scores rounded to **1 decimal place** |

**Data Flow:**

```
Data Source → Preprocessing → Feature Extraction → Momentum Calculation → State Determination → UI Rendering
```

---

## 3. Core Momentum Algorithm

### 3.1 Input Variables (Per Token, Rolling Window)

| Variable | Definition | Window |
|---|---|---|
| `p1m` | Price change % over last 60s | 60s |
| `p2m` | Price change % over last 120s | 120s |
| `v1m` | Volume in last 60s | 60s |
| `v1m_prev` | Volume in the 60s before that | 60–120s |
| `age_min` | Minutes since pool creation | — |
| `ath_ratio` | Current price ÷ ATH price | — |

### 3.2 Volume Acceleration

$$\text{VolAccel} = \frac{V_{\text{now}} - V_{\text{prev}}}{\max(V_{\text{prev}},\; \epsilon)}$$

> $\epsilon$ is a small constant to prevent division by zero.  
> **Clamping:** Result must be hard-clamped to $[-1.0,\; +1.5]$ before use in Momentum Score.

### 3.3 Momentum Score

$$M = 0.6 \times p1m + 0.4 \times p2m + 8 \times \text{VolAccel}_{\text{clamped}}$$

**Weight Philosophy:**
- **Price (60%):** Price is the ultimate expression of trend
- **Volume (40%):** Volume confirms trend authenticity

---

## 4. Color State Logic

The color state is the **visual core** of the terminal, driven entirely by Momentum Score and related conditions.

### 4.1 State Definitions

| Color | State | Entry Conditions | Exit Conditions |
|---|---|---|---|
| 🔵 **Blue** | Early Surge | $M \ge +6$ **AND** $p1m \ge +2.5\%$ **AND** $\text{VolAccel} \ge +0.30$ **AND** $\text{age\_min} \le 20$ | $\text{age\_min} > 25$ **OR** 2 consecutive ticks with $M < +4$ |
| 🟢 **Green** | Confirmed | 2 consecutive ticks: $M \ge +10$ **AND** $p2m \ge +4.0\%$ | Conditions no longer met for 2 ticks |
| 🟡 **Yellow** | Stalling | 2 consecutive ticks: $M \in [+4,\; +9]$ **OR** ($p1m \le 1.2\%$ **AND** $\text{VolAccel} \le 0.15$) | Conditions no longer met |
| 🔴 **Red** | Reversal | 2 consecutive ticks: $M \le -8$ | Conditions no longer met |

### 4.2 Panic Fast Rule (Exception)

> If $p1m \le -4.0\%$ **AND** $\text{VolAccel} \ge +0.25$ (high-volume crash), trigger Red in **1 tick only**.

### 4.3 UI Semantics

| Color | Trader Interpretation |
|---|---|
| 🔵 Blue | Cautiously optimistic — early attention |
| 🟢 Green | Trend confirmed — consider participation |
| 🟡 Yellow | Wait and see — direction unclear |
| 🔴 Red | Risk alert — consider exiting |

---

## 5. Anti-Flicker State Machine

The backend **must** maintain a `state_context` per active token (in memory / Redis).

### 5.1 Core Principle

```
1 hit  = noise    → no state change
2 hits = signal   → state transition fires
```

### 5.2 Pseudocode

```python
for each token:
    counter = get_counter(token, pending_state)

    if condition_met(pending_state):
        counter += 1
        if counter >= 2:
            change_state(token, pending_state)
            reset_counter(token)
    else:
        reset_counter(token)
```

### 5.3 Entry vs Exit Asymmetry

| Direction | Ticks Required | Rationale |
|---|---|---|
| **Entry** (Normal → Surging) | 2 consecutive | Filter noise |
| **Exit** (Surging → Normal) | 1 (immediate) | Protect capital |
| **Panic Red** | 1 (immediate) | Emergency protection |

---

## 6. Surge Detection System

### 6.1 Surge Score — Column Assignment

$$\text{Surge Score} = (V_{\text{volume}} \times W_{\text{vol}}) + (N_{\text{buys}} \times W_{\text{buy}}) + (N_{\text{wallets}} \times W_{\text{wallet}})$$

This absolute score is compared against the frontend Range Slider (default threshold: **31,000**).

**Column State Transitions:**

| Column | Entry Rule | Exit Rule |
|---|---|---|
| **Early** | Default. $\text{age\_min} \le 30$ + minimum volume | Momentum qualifies → Surging, or timeout |
| **Surging** | 2 consecutive ticks: Surge Score $\ge$ Range threshold | 1 tick: Score $< 80\%$ of threshold → immediate removal |
| **Graduated** | ATH broken ($\text{ath\_ratio} \ge 1.0$) + market cap milestone | **Permanent** — never demoted; color still updates |

### 6.2 Liquidity Surge

$$\text{Liquidity Multiplier} = \frac{\text{Current Liquidity}_{60s}}{\text{Average Liquidity}_{300s}}$$

| Transition | Threshold | Ticks |
|---|---|---|
| Enter Surge | Multiplier $\ge 1.2\times$ | 2 consecutive |
| Exit Surge | Multiplier $< 1.1\times$ | 1 (immediate) |

**States:** `Normal` → `Surging` → `Strong Surge`

### 6.3 Volume Surge

$$\text{Volume Multiplier} = \frac{\text{Current Volume}_{60s}}{\text{Average Volume}_{300s}}$$

| Transition | Threshold | Ticks |
|---|---|---|
| Enter Surge | Multiplier $\ge 1.3\times$ | 2 consecutive |
| Exit Surge | Multiplier $< 1.15\times$ | 1 (immediate) |

**States:** `Normal` → `Surging` → `Strong Surge`

> The hysteresis gap between entry and exit thresholds prevents edge-case flickering.

### 6.4 Surge Timing

Record the timestamp when Surge Score first crosses the threshold. `time_since_surge_seconds` = current time − that timestamp. If the token exits surge for > **120 seconds** (cooldown), the timer resets.

---

## 7. Risk & Holder Analysis

All risk metrics are computed server-side. The backend outputs a direct **UI color** — the frontend simply binds and renders (view–logic decoupling).

### 7.1 Definitions & Thresholds

| Metric | Definition | Display | 🟢 Safe | 🔴 Risk |
|---|---|---|---|---|
| **Top 10 Holders** | Top 10 wallets' combined holding ÷ total supply | Always | $\le 30\%$ | $> 30\%$ |
| **Dev Holding** | Creator wallet holding ÷ total supply | Always | $\le 10\%$ | $> 10\%$ |
| **Sniper Holding** | Wallets buying within 30s of pool creation (above min threshold) ÷ total supply | $\ge 3\%$ | $\le 10\%$ | $> 10\%$ |
| **Insider Holding** | Creator + pre-sale + creator-funded wallets ÷ total supply | Always | $\le 10\%$ | $> 10\%$ |
| **Bundle Holding** | $\ge 3$ wallets buying in same block (±1 block) ÷ total supply | $\ge 5\%$ | $\le 15\%$ | $> 15\%$ |

> **Note:** Insider labels have **permanent persistence** — once tagged, never removed.

### 7.2 Risk Interpretation

| Metric | Risk Meaning |
|---|---|
| Top 10 > 30% | Excessive chip concentration |
| Dev > 10% | Creator market control risk |
| Sniper > 10% | Bot dump risk |
| Insider > 10% | Insider trading risk |
| Bundle > 15% | Whale manipulation risk |

---

## 8. Price Action

$$\text{ATH Ratio} = \frac{P_{\text{current}}}{P_{\text{ATH}}}$$

- If current price is \$1.50 and ATH is \$1.00 → ratio = **1.5×**
- If current price is \$0.90 and ATH is \$1.00 → ratio = **0.9×**

> ATH Ratio also assists Yellow state detection: if $\text{ATH Ratio} \ge 0.94$ and momentum is declining, the token is likely stalling near its high.

---

## 9. API Specification (v1.0)

### 9.1 General Standards

| Item | Specification |
|---|---|
| Protocol | RESTful (JSON); WebSocket recommended for high-frequency push |
| Heartbeat | Server-side recalculation every 5 seconds |
| Formatting | Financial values in USD; percentages/scores to 1 decimal place |

### 9.2 Endpoint: Main Terminal Snapshot

Returns the comprehensive current state of a token.

**`GET /api/v1/terminal/{tokenAddress}/snapshot`**

```json
{
  "token_address": "8x5...sol",
  "symbol": "MEME",
  "timestamp": 1740706500,

  "basic_metrics": {
    "price_usd": 0.004512,
    "market_cap_usd": 4500000.0,
    "liquidity_usd": 850000.0,
    "volume_24h_usd": 1200000.0,
    "tx_time_last": "2026-02-28T01:35:00Z",
    "price_change_since_sync_pct": 3.2,
    "age_min": 15.5
  },

  "classification": {
    "column_state": "Surging",
    "color_state": "Green",
    "momentum_score": 14.2,
    "surge_score": 34200
  },

  "advanced_signals": {
    "liquidity_surge": {
      "score_multiplier": 1.6,
      "state": "Strong Surge"
    },
    "volume_surge": {
      "score_multiplier": 1.4,
      "state": "Surging"
    },
    "surge_timing": {
      "is_in_surge": true,
      "time_since_surge_seconds": 45
    }
  },

  "risk_analysis": {
    "top_10":      { "pct": 28.5, "ui_color": "Green" },
    "dev_holding": { "pct":  2.0, "ui_color": "Green" },
    "snipers":     { "pct":  4.5, "ui_color": "Green" },
    "insiders":    { "pct": 11.2, "ui_color": "Red"   },
    "bundle":      { "pct": 16.0, "ui_color": "Red"   }
  },

  "price_action": {
    "ath_price_usd": 0.003008,
    "ath_ratio": 1.5
  }
}
```

**Field Enumerations:**

| Field | Possible Values |
|---|---|
| `column_state` | `"Early"`, `"Surging"`, `"Graduated"` |
| `color_state` | `"Blue"`, `"Green"`, `"Yellow"`, `"Red"` |
| `liquidity_surge.state` | `"Normal"`, `"Surging"`, `"Strong Surge"` |
| `volume_surge.state` | `"Normal"`, `"Surging"`, `"Strong Surge"` |

### 9.3 Endpoint: Wallet Analysis (Drill-down)

Used when a user clicks a risk tag to view evidence.

**`GET /api/v1/terminal/{tokenAddress}/holders/analysis?type={BUNDLE|SNIPER|INSIDER}`**

```json
{
  "token_address": "8x5...sol",
  "type": "INSIDER",
  "wallets": [
    {
      "address": "Fk2...3pk",
      "holding_pct": 5.2,
      "reason": "Direct funding from Creator",
      "first_buy_time": "2026-02-28T01:00:05Z"
    }
  ]
}
```

---

## 10. Backend Logic Constraints

### 10.1 Volume Cleaning (Noise Filtering)

The backend **must** filter the following from all Volume and Momentum calculations:

| Filter | Rule |
|---|---|
| **Self-Trades (Wash Trading)** | Sender == Receiver, or circular patterns detected |
| **Zero-Value Transfers** | Airdrops or dust transactions with value < \$1.00 USD |

### 10.2 Data Continuity & Interpolation

If RPC data is missing for a specific 5-second tick, use **linear interpolation** from previous and next available data points.

> **Constraint:** Never return `null` or `NaN` to the frontend — this breaks chart rendering and signal calculations.

### 10.3 USD Conversion

$$P_{\text{USD}} = P_{\text{SOL}} \times \text{SOL/USD Oracle Price}$$

---

## 11. Technical Specifications

### 11.1 Default Configuration

```yaml
surging_range:        31000    # Surge threshold score
window_size:          60       # Rolling window (seconds)
update_interval:      5        # Update frequency (seconds)
hard_red_threshold:   -8       # Hard red momentum threshold
bundle_min_wallets:   3        # Minimum bundle wallet count
sniper_time_window:   30       # Sniper detection window (seconds)
```

### 11.2 Performance Targets

| Metric | Target |
|---|---|
| UI Update Latency | < 100ms |
| Backend Computation | < 500ms |
| Availability | 99.9% |

### 11.3 Sorting Rules

| Priority | Criterion | Order |
|---|---|---|
| Primary | Surge Score | Descending |
| Secondary | Surge Score rate of change | Descending |
| Tertiary | Latest activity timestamp | Descending |

---

## 12. Exception Handling

### 12.1 Edge Cases

| Scenario | Action |
|---|---|
| Zero-transaction tokens | Auto-exclude |
| Paused tokens | Auto-exclude |
| Newly created tokens (< 60s) | Optionally exclude |

### 12.2 Network Failures

| Scenario | Fallback |
|---|---|
| RPC timeout | Degrade to backup node |
| Missing data | Linear interpolation |
| API rate limiting | Request queuing |

---

## 13. Development & Testing

### 13.1 Debug Support

- **Momentum Score exposure** — available for debugging and A/B testing
- **Detailed logging** — record all state change trajectories
- **Simulation mode** — support historical data replay

### 13.2 Test Cases

| Category | Focus |
|---|---|
| Flicker test | Verify anti-flicker mechanism effectiveness |
| Boundary test | Test threshold edge-case behavior |
| Stress test | Simulate high-concurrency scenarios |

---

## 14. Risk Disclaimer

- All signals are for **reference only** and do not constitute investment advice
- Past performance does not guarantee future results
- Users bear all trading risks independently
- No insider trading information involved
- No specific buy/sell price targets provided
- Maintains technology-neutral stance

---

## 15. Roadmap

### Phase 1 (Current)

- [x] Core momentum detection
- [x] Color state system
- [x] Anti-flicker mechanism

### Phase 2 (Next)

- [ ] Social sentiment integration
- [ ] Machine learning optimization
- [ ] Mobile adaptation

### Phase 3 (Future)

- [ ] Cross-chain expansion
- [ ] Open API platform
- [ ] Quantitative strategy integration
