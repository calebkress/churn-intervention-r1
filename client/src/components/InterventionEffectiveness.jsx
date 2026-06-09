/**
 * InterventionEffectiveness.jsx
 * ==============================
 * Bar chart: retention rate per intervention type.
 * Data: all interventions from /api/interventions — aggregated client-side.
 * Shows which actions the PPO agent favored and how well they worked.
 */

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts'

const ACTION_LABELS = {
  do_nothing:             'Do nothing',
  send_email_offer:       'Email offer',
  outbound_call:          'Outbound call',
  discount_10pct:         '10% discount',
  escalate_to_retention:  'Escalate',
}

const tooltipStyle = {
  background: '#161B22',
  border: '1px solid #21262D',
  borderRadius: 6,
  color: '#E6EDF3',
  fontSize: 12,
}

export default function InterventionEffectiveness({ interventions }) {
  if (!interventions || interventions.length === 0) {
    return <Empty />
  }

  // Aggregate client-side: count retained/total per action type
  const counts = {}
  for (const iv of interventions) {
    const type = iv.type || 'unknown'
    if (!counts[type]) counts[type] = { retained: 0, total: 0 }
    counts[type].total++
    if (iv.outcome === 'retained') counts[type].retained++
  }

  const data = Object.entries(counts)
    .map(([type, { retained, total }]) => ({
      name: ACTION_LABELS[type] || type,
      retention: Math.round((retained / total) * 100),
      total,
    }))
    .sort((a, b) => b.retention - a.retention)

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#161B22" vertical={false} />
        <XAxis
          dataKey="name"
          tick={{ fill: '#7D8590', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          domain={[0, 100]}
          tickFormatter={v => `${v}%`}
          tick={{ fill: '#7D8590', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={36}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          formatter={(v, _, props) => [`${v}% (n=${props.payload.total})`, 'Retention rate']}
        />
        <Bar dataKey="retention" radius={[4, 4, 0, 0]}>
          {data.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.retention >= 70 ? '#2D7DD2' : entry.retention >= 50 ? '#2D7DD2' : '#F85149'}
            />
          ))}
        </Bar>
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