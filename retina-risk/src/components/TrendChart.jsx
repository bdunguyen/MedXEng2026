function formatPercent(value) {
  return `${Math.round(value * 100)}%`
}

function makePath(points, width, height, padding) {
  const values = points.length > 1 ? points : [0, points[0] ?? 0]

  return values
    .map((value, index) => {
      const x = padding + (index / (values.length - 1)) * (width - padding * 2)
      const y = height - padding - value * (height - padding * 2)
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
    })
    .join(' ')
}

export default function TrendChart({ cohort, onClose }) {
  if (!cohort) {
    return null
  }

  const width = 360
  const height = 180
  const padding = 28
  const trendPath = makePath(cohort.model.trend, width, height, padding)

  return (
    <div className="trend-overlay" role="dialog" aria-modal="true" aria-label={`${cohort.name} trend`}>
      <section className="trend-card">
        <header className="trend-header">
          <div>
            <p>trend::{cohort.model.risk_level}</p>
            <h2>{cohort.name}</h2>
          </div>
          <button aria-label="Close trend chart" onClick={onClose} type="button">
            x
          </button>
        </header>

        <svg className="trend-chart" viewBox={`0 0 ${width} ${height}`} role="img">
          <title>{cohort.name} six point risk trend</title>
          <line x1={padding} x2={width - padding} y1={height - padding} y2={height - padding} />
          <line x1={padding} x2={padding} y1={padding} y2={height - padding} />
          <path className="trend-line-glow" d={trendPath} />
          <path className="trend-line" d={trendPath} />
          {cohort.model.trend.map((value, index) => {
            const x = padding + (index / (cohort.model.trend.length - 1)) * (width - padding * 2)
            const y = height - padding - value * (height - padding * 2)

            return (
              <circle
                cx={x}
                cy={y}
                fill={cohort.color}
                key={`${value}-${index}`}
                r={4.5}
              />
            )
          })}
        </svg>

        <div className="trend-stats">
          <div>
            <span>Current</span>
            <strong>{formatPercent(cohort.model.risk_score)}</strong>
          </div>
          <div>
            <span>Burden</span>
            <strong>{formatPercent(cohort.disease_burden)}</strong>
          </div>
          <div>
            <span>Patients</span>
            <strong>{cohort.population}</strong>
          </div>
        </div>
      </section>
    </div>
  )
}
