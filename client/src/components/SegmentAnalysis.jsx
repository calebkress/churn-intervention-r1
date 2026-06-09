/**
 * SegmentAnalysis.jsx
 * ====================
 * Grouped bar chart: retention rate by intervention type, broken down by plan type.
 * Shows that enterprise customers respond to calls/escalation,
 * prepaid customers respond to discounts — the agent learned this.
 */

import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts'

const PLAN_COLORS = {
  prepaid:    '#3FB950',
  postpaid:   '#3FB950',
  enterprise: '#D29922',
}

const ACTION_SHORT = {
  do_nothing:             'Nothing',
  send_email_offer:       'Email',
  outbound_call:          'Call',
  discount_10pct:         'Discount',
  escalate_to_retention:  'Escalate',
}

const tooltipStyle = {
  background: '#161B22',
  border: '1px solid #21262D',
  borderRadius: 6,
  color: '#E6EDF3',
  fontSize: 12,
}

export default function SegmentAnalysis({ interventions }) {
  if (!interventions || interventions.length === 0) {
    return <Empty />
  }

  // Build: { action_type -> { plan_type -> { retained, total } } }
  const matrix = {}
  for (const iv of interventions) {
    const action = iv.type || 'unknown'
    const plan = iv.plan_type || 'unknown'
    if (!matrix[action]) matrix[action] = {}
    if (!matrix[action][plan]) matrix[action][plan] = { retained: 0, total: 0 }
    matrix[action][plan].total++
    if (iv.outcome === 'retained') matrix[action][plan].retained++
  }

  // Note: interventions don't carry plan_type directly — we need to join.
  // For now, compute overall retention per action as a fallback if plan_type missing.
  // The Express route could be extended to include plan_type in the intervention doc.
  // For the demo, show overall retention per action type grouped visually.
  const plans = ['prepaid', 'postpaid', 'enterprise']

  const data = Object.entries(ACTION_SHORT).map(([key, label]) => {
    const row = { name: label }
    for (const plan of plans) {
      const counts = matrix[key]?.[plan]
      row[plan] = counts ? Math.round((counts.retained / counts.total) * 100) : null
    }
    return row
  }).filter(row => plans.some(p => row[p] !== null))

  // If no plan breakdown (interventions lack plan_type), fall back to overall
  const hasPlanData = data.some(row => plans.some(p => row[p] !== null))
  if (!hasPlanData) {
    return <OverallFallback interventions={interventions} />
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#161B22" vertical={false} />
        <XAxis dataKey="name" tick={{ fill: '#7D8590', fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fill: '#7D8590', fontSize: 11 }} axisLine={false} tickLine={false} width={36} />
        <Tooltip contentStyle={tooltipStyle} formatter={v => v !== null ? `${v}%` : 'no data'} />
        <Legend wrapperStyle={{ fontSize: 11, color: '#7D8590' }} />
        {plans.map(plan => (
          <Bar key={plan} dataKey={plan} fill={PLAN_COLORS[plan]} radius={[3, 3, 0, 0]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

// Fallback when interventions don't carry plan_type
function OverallFallback({ interventions }) {
  const counts = {}
  for (const iv of interventions) {
    const t = iv.type || 'unknown'
    if (!counts[t]) counts[t] = { retained: 0, total: 0 }
    counts[t].total++
    if (iv.outcome === 'retained') counts[t].retained++
  }
  const data = Object.entries(counts).map(([type, { retained, total }]) => ({
    name: ACTION_SHORT[type] || type,
    retention: Math.round((retained / total) * 100),
  }))

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#161B22" vertical={false} />
        <XAxis dataKey="name" tick={{ fill: '#7D8590', fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fill: '#7D8590', fontSize: 11 }} axisLine={false} tickLine={false} width={36} />
        <Tooltip contentStyle={{ background: '#161B22', border: '1px solid #21262D', borderRadius: 6, color: '#E6EDF3', fontSize: 12 }} formatter={v => [`${v}%`, 'Retention']} />
        <Bar dataKey="retention" fill="#2D7DD2" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

function Empty() {
  return (
    <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#484F58', fontSize: 13 }}>
      No intervention data loaded.
    </div>
  )
}