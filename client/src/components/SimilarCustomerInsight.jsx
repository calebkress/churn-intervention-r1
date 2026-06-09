/**
 * SimilarCustomerInsight.jsx
 * ==========================
 *
 * Data flow: this component → Express /api/ml/insights/:id → FastAPI /insights/:id
 *            → lc/insights.py → Atlas $vectorSearch + gpt-4o-mini
 */

import { useState } from 'react'

const ACTION_LABELS = {
  do_nothing:             'Do nothing',
  send_email_offer:       'Email offer',
  outbound_call:          'Outbound call',
  discount_10pct:         '10% discount',
  escalate_to_retention:  'Escalate',
}

const styles = {
  layout: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: 24,
  },
  left: {},
  right: {},
  controlRow: {
    display: 'flex',
    gap: 10,
    marginBottom: 16,
    alignItems: 'center',
  },
  select: {
    flex: 1,
    background: '#161B22',
    border: '1px solid #21262D',
    borderRadius: 6,
    color: '#E6EDF3',
    padding: '8px 12px',
    fontSize: 13,
  },
  btn: {
    background: '#3FB950',
    color: '#0D1117',
    border: 'none',
    borderRadius: 6,
    padding: '8px 18px',
    fontSize: 13,
    fontWeight: 500,
    whiteSpace: 'nowrap',
  },
  btnDisabled: {
    background: '#161B22',
    color: '#484F58',
    border: 'none',
    borderRadius: 6,
    padding: '8px 18px',
    fontSize: 13,
    fontWeight: 500,
    whiteSpace: 'nowrap',
    cursor: 'not-allowed',
  },
  insightBox: {
    background: '#0D1117',
    border: '1px solid #21262D',
    borderRadius: 8,
    padding: '16px 18px',
    fontSize: 14,
    lineHeight: 1.7,
    color: '#cbd5e1',
    minHeight: 100,
  },
  insightLoading: {
    color: '#484F58',
    fontStyle: 'italic',
    fontSize: 13,
  },
  customerCard: {
    background: '#0D1117',
    border: '1px solid #161B22',
    borderRadius: 8,
    padding: '12px 16px',
    marginBottom: 10,
    fontSize: 12,
  },
  customerHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: 6,
    color: '#7D8590',
    fontWeight: 600,
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
  },
  similarityBadge: {
    background: '#1C2128',
    color: '#3FB950',
    borderRadius: 4,
    padding: '1px 6px',
    fontSize: 11,
  },
  interventionList: {
    marginTop: 6,
    display: 'flex',
    flexWrap: 'wrap',
    gap: 4,
  },
  ivBadge: {
    fontSize: 10,
    padding: '2px 6px',
    borderRadius: 4,
    fontWeight: 500,
  },
  retained: { background: '#1C2128', color: '#3FB950' },
  churned:  { background: '#1A0A09', color: '#F85149' },
  label: {
    fontSize: 11,
    color: '#7D8590',
    marginBottom: 8,
    textTransform: 'uppercase',
    letterSpacing: '0.06em',
  },
  errorBox: {
    background: '#1A0A09',
    border: '1px solid #3D0E0D',
    borderRadius: 8,
    padding: '12px 16px',
    fontSize: 13,
    color: '#F85149',
  },
}

export default function SimilarCustomerInsight({ customers }) {
  const [selectedId, setSelectedId] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function fetchInsight() {
    if (!selectedId) return
    setLoading(true)
    setResult(null)
    setError(null)
    try {
      const res = await fetch(`/api/ml/insights/${selectedId}`)
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || err.error || `HTTP ${res.status}`)
      }
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Show high-risk customers first in selector
  const sortedCustomers = [...(customers || [])].sort(
    (a, b) => (b.churn_probability || 0) - (a.churn_probability || 0)
  )

  return (
    <div style={styles.layout}>
      {/* Left: controls + insight */}
      <div style={styles.left}>
        <div style={styles.controlRow}>
          <select
            style={styles.select}
            value={selectedId}
            onChange={e => { setSelectedId(e.target.value); setResult(null); setError(null) }}
          >
            <option value="">Select a customer…</option>
            {sortedCustomers.map(c => (
              <option key={c.customer_id} value={c.customer_id}>
                {c.customer_id.slice(0, 8)}… — {c.plan_type} — {((c.churn_probability || 0) * 100).toFixed(0)}% churn risk
              </option>
            ))}
          </select>
          <button
            style={loading || !selectedId ? styles.btnDisabled : styles.btn}
            onClick={fetchInsight}
            disabled={loading || !selectedId}
          >
            {loading ? 'Thinking…' : 'Get insight'}
          </button>
        </div>

        <div style={styles.label}>LangChain recommendation</div>
        <div style={styles.insightBox}>
          {loading && <span style={styles.insightLoading}>Querying Atlas Vector Search and generating recommendation…</span>}
          {error && <div style={styles.errorBox}>{error}</div>}
          {result && !loading && result.insight}
          {!loading && !result && !error && (
            <span style={styles.insightLoading}>
              Select a customer and click "Get insight" to run Atlas Vector Search + LangChain.
            </span>
          )}
        </div>
      </div>

      {/* Right: similar customers */}
      <div style={styles.right}>
        <div style={styles.label}>
          Similar customers from Atlas Vector Search
          {result && ` (${result.similar_customers?.length || 0} found)`}
        </div>
        {result?.similar_customers?.map((c, i) => (
          <div key={c.customer_id} style={styles.customerCard}>
            <div style={styles.customerHeader}>
              <span>
                {c.customer_id.slice(0, 8)}… · {c.plan_type} · {c.tenure_months}mo tenure
              </span>
              <span style={styles.similarityBadge}>
                {((c.similarity_score || 0) * 100).toFixed(1)}% match
              </span>
            </div>
            <div style={{ color: '#7D8590', fontSize: 11, marginBottom: 4 }}>
              Churn risk: {((c.churn_probability || 0) * 100).toFixed(0)}%
            </div>
            <div style={styles.interventionList}>
              {c.intervention_history?.length === 0 && (
                <span style={{ color: '#484F58', fontSize: 11 }}>No interventions on record</span>
              )}
              {c.intervention_history?.slice(0, 8).map((iv, j) => (
                <span
                  key={j}
                  style={{
                    ...styles.ivBadge,
                    ...(iv.outcome === 'retained' ? styles.retained : styles.churned),
                  }}
                >
                  {ACTION_LABELS[iv.type] || iv.type} → {iv.outcome}
                </span>
              ))}
              {(c.intervention_history?.length || 0) > 8 && (
                <span style={{ color: '#484F58', fontSize: 10 }}>
                  +{c.intervention_history.length - 8} more
                </span>
              )}
            </div>
          </div>
        ))}
        {!result && !loading && (
          <div style={{ color: '#1C2128', fontSize: 13, paddingTop: 8 }}>
            Similar customers will appear here after running an insight query.
          </div>
        )}
      </div>
    </div>
  )
}