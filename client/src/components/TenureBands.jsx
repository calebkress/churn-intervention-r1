/**
 * TenureBands.jsx
 * ================
 * Bar chart: churn rate by customer tenure band (0-12mo, 13-24mo, etc.)
 * Shows that newer customers churn more — the agent learned to prioritize them.
 * Data: customers list for churn_probability, bucketed by tenure_months.
 */

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts'

const BANDS = [
  { label: '0–12mo',  min: 0,   max: 12  },
  { label: '13–24mo', min: 13,  max: 24  },
  { label: '25–48mo', min: 25,  max: 48  },
  { label: '49–72mo', min: 49,  max: 72  },
  { label: '73–120mo',min: 73,  max: 120 },
]

const tooltipStyle = {
  background: '#1e2433',
  border: '1px solid #2d3748',
  borderRadius: 6,
  color: '#e2e8f0',
  fontSize: 12,
}

export default function TenureBands({ customers }) {
  if (!customers || customers.length === 0) {
    return <Empty />
  }

  const data = BANDS.map(band => {
    const group = customers.filter(
      c => c.tenure_months >= band.min && c.tenure_months <= band.max
    )
    if (group.length === 0) return { name: band.label, avgChurn: 0, count: 0 }

    const avgChurn = group.reduce((sum, c) => sum + (c.churn_probability || 0), 0) / group.length
    return {
      name: band.label,
      avgChurn: Math.round(avgChurn * 100),
      count: group.length,
    }
  })

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2433" vertical={false} />
        <XAxis
          dataKey="name"
          tick={{ fill: '#64748b', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          domain={[0, 100]}
          tickFormatter={v => `${v}%`}
          tick={{ fill: '#64748b', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={36}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          formatter={(v, _, props) => [`${v}% avg (n=${props.payload.count})`, 'Churn probability']}
        />
        <Bar dataKey="avgChurn" radius={[4, 4, 0, 0]}>
          {data.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.avgChurn >= 50 ? '#f87171' : entry.avgChurn >= 35 ? '#f59e0b' : '#34d399'}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function Empty() {
  return (
    <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569', fontSize: 13 }}>
      No customer data loaded.
    </div>
  )
}