import { useEffect, useState } from 'react'

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

function makeTimelineLabels(count) {
  if (count <= 1) {
    return ['+12m']
  }

  return Array.from({ length: count }, (_, index) => {
    const month = Math.round((index / (count - 1)) * 12)

    return index === 0 ? 'now' : `+${month}m`
  })
}

export default function TrendChart({ cohort, datasetKey, onClose }) {
  const [insight, setInsight] = useState(null)
  const [insightError, setInsightError] = useState('')
  const [isInsightLoading, setIsInsightLoading] = useState(false)

  useEffect(() => {
    if (!cohort) {
      return
    }

    const controller = new AbortController()

    async function loadInsight() {
      setInsight(null)
      setInsightError('')
      setIsInsightLoading(true)

      try {
        const response = await fetch('/api/insights', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ cohortId: cohort.id, datasetKey }),
          signal: controller.signal,
        })

        const contentType = response.headers.get('content-type') || ''
        if (!contentType.includes('application/json')) {
          throw new Error('Insights endpoint unavailable in this environment.')
        }

        const payload = await response.json()

        if (!response.ok) {
          throw new Error(payload.error || 'Unable to generate insight.')
        }

        setInsight(payload)
      } catch (error) {
        if (error.name !== 'AbortError') {
          setInsightError(error.message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsInsightLoading(false)
        }
      }
    }

    loadInsight()

    return () => controller.abort()
  }, [cohort, datasetKey])

  if (!cohort) {
    return null
  }

  const width = 360
  const height = 180
  const padding = 28
  const trendPath = makePath(cohort.model.trend, width, height, padding)
  const timelineLabels = makeTimelineLabels(cohort.model.trend.length)
  const latestRisk = cohort.model.trend[cohort.model.trend.length - 1]

  return (
    <div className="trend-overlay" role="dialog" aria-modal="true" aria-label={`${cohort.name} trend`}>
      <section className="trend-card">
        <header className="trend-header">
          <div>
            <p>12-month admission forecast::{cohort.model.risk_level}</p>
            <h2>{cohort.name}</h2>
          </div>
          <button aria-label="Close trend chart" onClick={onClose} type="button">
            x
          </button>
        </header>

        <svg className="trend-chart" viewBox={`0 0 ${width} ${height + 16}`} role="img">
          <title>{cohort.name} projected admission risk over the next 12 months</title>
          <line x1={padding} x2={width - padding} y1={height - padding} y2={height - padding} />
          <line x1={padding} x2={padding} y1={padding} y2={height - padding} />
          <text className="axis-label" transform="rotate(-90)" x={-(height / 2)} y={10} textAnchor="middle">
            Admission Risk
          </text>
          <text className="axis-label" x={width / 2} y={height + 13} textAnchor="middle">
            Time
          </text>
          <text x={padding - 8} y={padding + 4} textAnchor="end">
            100%
          </text>
          <text x={padding - 8} y={height / 2 + 4} textAnchor="end">
            50%
          </text>
          <text x={padding - 8} y={height - padding + 4} textAnchor="end">
            0%
          </text>
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
          {timelineLabels.map((label, index) => {
            const x = padding + (index / (timelineLabels.length - 1)) * (width - padding * 2)

            return (
              <text className="trend-tick-label" key={label} x={x} y={height - 6} textAnchor="middle">
                {label}
              </text>
            )
          })}
        </svg>

        <div className="trend-context">
          <strong>Hackathon task lens</strong>
          <p>
            This timeline frames the cohort as a next-12-month hospital admission forecast. The
            final checkpoint is the model&apos;s held-out-style risk readout for prioritizing review.
          </p>
        </div>

        <div className="trend-stats">
          <div>
            <span>Current</span>
            <strong>{formatPercent(cohort.model.risk_score)}</strong>
          </div>
          <div>
            <span>12 mo</span>
            <strong>{formatPercent(latestRisk)}</strong>
          </div>
          <div>
            <span>AUROC</span>
            <strong>{cohort.auroc != null ? cohort.auroc.toFixed(3) : 'n/a'}</strong>
          </div>
          <div>
            <span>Patients</span>
            <strong>{cohort.population}</strong>
          </div>
        </div>

        <div className="trend-insights" aria-live="polite">
          <div className="trend-insights-header">
            <h3>Cohort insight</h3>
            {insight?.source === 'gemini' && <span>Gemini</span>}
          </div>
          {isInsightLoading && <p>Generating cohort insight...</p>}
          {insightError && <p className="trend-insights-error">{insightError}</p>}
          {insight && (
            <>
              <p>{insight.summary}</p>
              <p>{insight.trendInterpretation}</p>
              <p>{insight.chartCallout}</p>
              <div>
                <strong>Risk drivers</strong>
                <ul>
                  {insight.riskDrivers.map((driver) => (
                    <li key={driver}>{driver}</li>
                  ))}
                </ul>
              </div>
              <div>
                <strong>Caveats</strong>
                <ul>
                  {insight.caveats.map((caveat) => (
                    <li key={caveat}>{caveat}</li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </div>
      </section>
    </div>
  )
}
