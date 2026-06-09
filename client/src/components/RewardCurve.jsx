/**
 * RewardCurve.jsx
 * ===============
 * Shows agent mean reward over training timesteps.
 * Data: training_runs.reward_curve array — [{step, mean_reward}, ...]
 * An upward trend = agent learning to retain customers more efficiently.
 */

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

const tooltipStyle = {
  background: '#161B22',
  border: '1px solid #21262D',
  borderRadius: 6,
  color: '#E6EDF3',
  fontSize: 12,
}

export default function RewardCurve({ rewardCurve }) {
  if (!rewardCurve || rewardCurve.length === 0) {
    return <Empty message="No reward curve data for this run." />
  }

  // Sample down if very large — recharts handles ~200 points well
  const data = rewardCurve.length > 200
    ? rewardCurve.filter((_, i) => i % Math.ceil(rewardCurve.length / 200) === 0)
    : rewardCurve

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#161B22" />
        <XAxis
          dataKey="step"
          tickFormatter={v => `${(v / 1000).toFixed(0)}k`}
          tick={{ fill: '#7D8590', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: '#7D8590', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={36}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          labelFormatter={v => `Step ${v.toLocaleString()}`}
          formatter={v => [v.toFixed(2), 'Mean reward']}
        />
        <Line
          type="monotone"
          dataKey="mean_reward"
          stroke="#3FB950"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: '#2D7DD2' }}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

function Empty({ message }) {
  return (
    <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#484F58', fontSize: 13 }}>
      {message}
    </div>
  )
}