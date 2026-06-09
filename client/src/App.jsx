import { useState, useEffect } from 'react'
import RewardCurve from './components/RewardCurve'
import InterventionEffectiveness from './components/InterventionEffectiveness'
import SegmentAnalysis from './components/SegmentAnalysis'
import TenureBands from './components/TenureBands'
import OverIntervention from './components/OverIntervention'
import SimilarCustomerInsight from './components/SimilarCustomerInsight'

// Canonical run — highlighted in the training runs selector
const CANONICAL_RUN_ID = 'b4193c90'

const MG = {
  bg:          '#0D1117',
  card:        '#161B22',
  cardAlt:     '#21262D',
  border:      '#21262D',
  green:       '#3FB950',
  blue:        '#2D7DD2',
  greenCta:    '#3FB950',
  textPrimary: '#E6EDF3',
  textMuted:   '#7D8590',
  textDim:     '#484F58',
  red:         '#F85149',
  amber:       '#D29922',
}

const styles = {
  app: {
    maxWidth: 1400,
    margin: '0 auto',
    padding: '24px 32px',
  },
  header: {
    marginBottom: 32,
    borderBottom: `1px solid ${MG.border}`,
    paddingBottom: 20,
    display: 'flex',
    alignItems: 'baseline',
    gap: 12,
  },
  title: {
    fontSize: 22,
    fontWeight: 700,
    color: MG.textPrimary,
  },
  subtitle: {
    fontSize: 13,
    color: MG.textMuted,
    marginLeft: 'auto',
  },
  badge: {
    display: 'inline-block',
    background: MG.green,
    color: MG.bg,
    fontSize: 11,
    fontWeight: 700,
    padding: '2px 8px',
    borderRadius: 4,
    letterSpacing: '0.04em',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: 16,
    marginBottom: 16,
  },
  gridFull: {
    display: 'grid',
    gridTemplateColumns: '1fr',
    gap: 16,
    marginBottom: 16,
  },
  card: {
    background: MG.card,
    border: `1px solid ${MG.border}`,
    borderRadius: 10,
    padding: '20px 24px',
  },
  cardTitle: {
    fontSize: 11,
    fontWeight: 700,
    color: MG.green,
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    marginBottom: 16,
  },
  statsRow: {
    display: 'flex',
    gap: 12,
    marginBottom: 20,
  },
  stat: {
    flex: 1,
    background: MG.card,
    border: `1px solid ${MG.border}`,
    borderRadius: 10,
    padding: '16px 20px',
  },
  statLabel: {
    fontSize: 11,
    color: MG.textMuted,
    textTransform: 'uppercase',
    letterSpacing: '0.06em',
    marginBottom: 6,
  },
  statValue: {
    fontSize: 28,
    fontWeight: 700,
    color: MG.textPrimary,
  },
  statSub: {
    fontSize: 12,
    color: MG.textDim,
    marginTop: 2,
  },
  statGreen: { color: MG.green },
  statRed:   { color: MG.red },
  customerPanel: {
    display: 'flex',
    gap: 12,
    marginBottom: 20,
    alignItems: 'center',
  },
  select: {
    flex: 1,
    background: MG.cardAlt,
    border: `1px solid ${MG.border}`,
    borderRadius: 6,
    color: MG.textPrimary,
    padding: '8px 12px',
    fontSize: 13,
  },
  btn: {
    background: MG.green,
    color: MG.bg,
    border: 'none',
    borderRadius: 6,
    padding: '8px 16px',
    fontSize: 13,
    fontWeight: 700,
  },
  sectionLabel: {
    fontSize: 11,
    color: MG.green,
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    marginBottom: 12,
    marginTop: 8,
  },
}

export default function App() {
  const [trainingRun, setTrainingRun] = useState(null)
  const [allRuns, setAllRuns] = useState([])
  const [interventions, setInterventions] = useState([])
  const [customers, setCustomers] = useState([])
  const [selectedCustomer, setSelectedCustomer] = useState(null)
  const [loading, setLoading] = useState(true)

  // Load everything on mount
  useEffect(() => {
    async function loadAll() {
      try {
        const [runsRes, intRes, custRes] = await Promise.all([
          fetch('/api/training-runs'),
          fetch('/api/interventions?limit=5000'),
          fetch('/api/customers?limit=200&page=1'),
        ])
        const runsData = await runsRes.json()
        const intData  = await intRes.json()
        const custData = await custRes.json()

        setAllRuns(runsData.runs || [])
        setInterventions(intData.interventions || [])
        setCustomers(custData.customers || [])

        // Default to canonical run
        const canonical = (runsData.runs || []).find(r =>
          r.run_id.startsWith(CANONICAL_RUN_ID)
        ) || runsData.runs?.[0]

        if (canonical) {
          // Fetch full run (includes reward_curve)
          const fullRes = await fetch(`/api/training-runs/${canonical.run_id}`)
          const fullRun = await fullRes.json()
          setTrainingRun(fullRun)
        }
      } catch (err) {
        console.error('Failed to load dashboard data:', err)
      } finally {
        setLoading(false)
      }
    }
    loadAll()
  }, [])

  async function handleRunChange(e) {
    const runId = e.target.value
    const res = await fetch(`/api/training-runs/${runId}`)
    const run = await res.json()
    setTrainingRun(run)
  }

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', color: '#64748b' }}>
        Loading dashboard…
      </div>
    )
  }

  const churnReduction = trainingRun
    ? ((trainingRun.churn_rate_reduction || 0) * 100).toFixed(1)
    : '—'
  const baselineRate = trainingRun
    ? ((trainingRun.churn_rate_baseline || 0) * 100).toFixed(1)
    : '—'
  const trainedRate = trainingRun
    ? ((trainingRun.churn_rate_trained || 0) * 100).toFixed(1)
    : '—'

  return (
    <div style={styles.app}>
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.title}>Churn Intervention RL</div>
        <span style={styles.badge}>MongoDB Atlas</span>
        <div style={styles.subtitle}>
          PPO agent · {interventions.length.toLocaleString()} interventions · 10,000 customers
        </div>
      </div>

      {/* Run selector */}
      <div style={styles.customerPanel}>
        <select style={styles.select} onChange={handleRunChange}
          value={trainingRun?.run_id || ''}>
          {allRuns.map(r => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_id.slice(0, 8)} — {r.algorithm} — {((r.churn_rate_reduction || 0) * 100).toFixed(1)}pt reduction
              {r.run_id.startsWith(CANONICAL_RUN_ID) ? ' ★ canonical' : ''}
            </option>
          ))}
        </select>
      </div>

      {/* Key stats */}
      <div style={styles.statsRow}>
        <div style={styles.stat}>
          <div style={styles.statLabel}>Baseline churn</div>
          <div style={{...styles.statValue, ...styles.statRed}}>{baselineRate}%</div>
          <div style={styles.statSub}>before intervention</div>
        </div>
        <div style={styles.stat}>
          <div style={styles.statLabel}>Trained churn</div>
          <div style={{...styles.statValue, ...styles.statGreen}}>{trainedRate}%</div>
          <div style={styles.statSub}>with PPO policy</div>
        </div>
        <div style={styles.stat}>
          <div style={styles.statLabel}>Churn reduction</div>
          <div style={{...styles.statValue, ...styles.statGreen}}>{churnReduction}pts</div>
          <div style={styles.statSub}>absolute improvement</div>
        </div>
        <div style={styles.stat}>
          <div style={styles.statLabel}>Mean reward</div>
          <div style={styles.statValue}>{trainingRun?.mean_eval_reward?.toFixed(1) || '—'}</div>
          <div style={styles.statSub}>per episode</div>
        </div>
        <div style={styles.stat}>
          <div style={styles.statLabel}>Customers</div>
          <div style={styles.statValue}>{(trainingRun?.n_customers || 0).toLocaleString()}</div>
          <div style={styles.statSub}>in Atlas pool</div>
        </div>
      </div>

      {/* Charts row 1 */}
      <div style={styles.grid}>
        <div style={styles.card}>
          <div style={styles.cardTitle}>Agent learning curve</div>
          <RewardCurve rewardCurve={trainingRun?.reward_curve || []} />
        </div>
        <div style={styles.card}>
          <div style={styles.cardTitle}>Intervention effectiveness</div>
          <InterventionEffectiveness interventions={interventions} />
        </div>
      </div>

      {/* Charts row 2 */}
      <div style={styles.grid}>
        <div style={styles.card}>
          <div style={styles.cardTitle}>Effectiveness by segment</div>
          <SegmentAnalysis interventions={interventions} />
        </div>
        <div style={styles.card}>
          <div style={styles.cardTitle}>Churn by tenure band</div>
          <TenureBands interventions={interventions} customers={customers} />
        </div>
      </div>

      {/* Over-intervention full width */}
      <div style={styles.gridFull}>
        <div style={styles.card}>
          <div style={styles.cardTitle}>Over-intervention analysis</div>
          <OverIntervention interventions={interventions} />
        </div>
      </div>

      {/* LangChain insight — the demo moment */}
      <div style={styles.sectionLabel}>Atlas Vector Search + LangChain</div>
      <div style={styles.gridFull}>
        <div style={styles.card}>
          <div style={styles.cardTitle}>Similar customer insight</div>
          <SimilarCustomerInsight customers={customers} />
        </div>
      </div>
    </div>
  )
}