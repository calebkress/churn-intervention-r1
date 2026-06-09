/**
 * OverIntervention.jsx
 * =====================
 * Shows customers who were contacted more than 3 times and their outcomes.
 * The reward function penalizes over-contacting — this chart shows the agent
 * learned to avoid it (most customers have ≤3 interventions).
 */

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts'

const tooltipStyle = {
  background: '#161B22',
  border: '1px solid #21262D',
  borderRadius: 6,
  color: '#E6EDF3',
  fontSize: 12,
}

export default function OverIntervention({ interventions }) {
  if (!interventions || interventions.length === 0) {
    return <Empty />
  }

  // Count interventions per customer
  const countByCustomer = {}
  const outcomeByCustomer = {}
  for (const iv of interventions) {
    const id = iv.customer_id
    countByCustomer[id] = (countByCustomer[id] || 0) + 1
    // Last outcome wins — rough approximation
    outcomeByCustomer[id] = iv.outcome
  }

  // Bucket by intervention count
  const buckets = {}
  for (const [id, count] of Object.entries(countByCustomer)) {
    const bucket = count > 8 ? '9+' : String(count)
    if (!buckets[bucket]) buckets[bucket] = { retained: 0, churned: 0 }
    const outcome = outcomeByCustomer[id]
    if (outcome === 'retained') buckets[bucket].retained++
    else if (outcome === 'churned') buckets[bucket].churned++
  }

  const ORDER = ['1','2','3','4','5','6','7','8','9+']
  const data = ORDER
    .filter(k => buckets[k])
    .map(k => ({
      name: `${k}×`,
      retained: buckets[k].retained,
      churned: buckets[k].churned,
      overContact: parseInt(k) > 3 || k === '9+',
    }))

  return (
    <div>
      <div style={{ fontSize: 12, color: '#7D8590', marginBottom: 12 }}>
        Reward function penalizes {'>'} 3 contacts per episode. Agent learned to avoid over-contacting low-risk customers.
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#161B22" vertical={false} />
          <XAxis
            dataKey="name"
            tick={{ fill: '#7D8590', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            label={{ value: 'Interventions per customer', position: 'insideBottom', offset: -2, fill: '#484F58', fontSize: 11 }}
          />
          <YAxis
            tick={{ fill: '#7D8590', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={36}
          />
          <Tooltip contentStyle={tooltipStyle} />
          <Legend wrapperStyle={{ fontSize: 11, color: '#7D8590', paddingTop: 8 }} />
          <Bar dataKey="retained" stackId="a" fill="#2D7DD2" radius={[0, 0, 0, 0]} name="Retained" />
          <Bar dataKey="churned"  stackId="a" fill="#F85149" radius={[4, 4, 0, 0]} name="Churned" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function Empty() {
  return (
    <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#484F58', fontSize: 13 }}>
      No intervention data loaded.
    </div>
  )
}