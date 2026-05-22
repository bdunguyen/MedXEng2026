function formatPercent(value) {
  return `${Math.round(value * 100)}%`
}

function getTrendLabel(cohort) {
  const trend = cohort.model.trend

  if (trend.length < 2) {
    return 'Pulse shows cohort activity; trend data is limited.'
  }

  const first = trend[0]
  const last = trend[trend.length - 1]
  const direction = last > first ? 'worsening' : 'stable or improving'

  return `Pulse shows trend direction: ${direction} from ${formatPercent(first)} to ${formatPercent(last)}.`
}

function getDepthLabel(cohort) {
  const depthPercent = formatPercent(cohort.normalized_disease_burden)
  const basis = cohort.depth_basis ?? 'disease burden'

  return `Center distance uses normalized ${basis}: ${depthPercent}. Closer to center means higher burden or risk in this dataset.`
}

export default function SidePanel({
  cohorts,
  datasetKey,
  datasetOptions,
  metadata,
  onDatasetChange,
  onSelectCohort,
  onShowChart,
  selectedCohort,
}) {
  const sortedCohorts = [...cohorts].sort(
    (a, b) => b.normalized_disease_burden - a.normalized_disease_burden,
  )

  function handleCohortClick(cohort) {
    onSelectCohort(cohort)
    if (onShowChart) {
      onShowChart(cohort)
    }
  }
  return (
    <aside className="retina-panel">
      <div className="project-signature" aria-label="MedXEng2026 Britney and Leo">
        <span>MedXEng2026</span>
        <span>Britney &amp; Leo</span>
      </div>
      <div className="dataset-switch" aria-label="Prediction dataset switch">
        {Object.entries(datasetOptions).map(([key, dataset]) => (
          <button
            aria-pressed={datasetKey === key}
            className={datasetKey === key ? 'active' : ''}
            key={key}
            onClick={() => onDatasetChange(key)}
            type="button"
          >
            {dataset.label}
          </button>
        ))}
      </div>
      <div className="prediction-meta">
        <p>{metadata.project}</p>
        <p>{metadata.prediction_target}</p>
        <p>
          model::{metadata.default_model} · validation n::{metadata.validation_n ?? 'test'} · auroc::
          {metadata.validation_auroc ?? 'n/a'}
        </p>
      </div>
      <div className="finding-readout cohort-readout">
        <span style={{ backgroundColor: selectedCohort.color }} />
        <div>
          <h2>{selectedCohort.name}</h2>
          <p>
            {formatPercent(selectedCohort.model.risk_score)} {selectedCohort.model.risk_level} risk
            · {selectedCohort.population} patients
          </p>
        </div>
      </div>
      <div className="risk-meter" aria-label={`${selectedCohort.name} risk score`}>
        <div style={{ width: formatPercent(selectedCohort.model.risk_score) }} />
      </div>
      <div className="node-legend" aria-label={`${selectedCohort.name} node feature explanation`}>
        <h3>Node features</h3>
        <div>
          <strong>Depth</strong>
          <p>{getDepthLabel(selectedCohort)}</p>
        </div>
        <div>
          <strong>Pulse</strong>
          <p>{getTrendLabel(selectedCohort)}</p>
        </div>
        <div>
          <strong>Color</strong>
          <p>Color uses a distinct cohort hue, with model risk shifting the tint and intensity.</p>
        </div>
      </div>
      <div className="feature-list">
        <h3>Top model features</h3>
        {selectedCohort.model.top_features.map((item) => (
          <div className="feature-row" key={item.feature}>
            <span>{item.feature}</span>
            <strong>{formatPercent(item.importance)}</strong>
          </div>
        ))}
      </div>
      <div className="cohort-list" aria-label="Prediction cohorts">
        {sortedCohorts.map((cohort) => (
          <button
            className={cohort.id === selectedCohort.id ? 'active' : ''}
            key={cohort.id}
            onClick={() => handleCohortClick(cohort)}
            type="button"
          >
            <span style={{ backgroundColor: cohort.color }} />
            {cohort.name}
          </button>
        ))}
      </div>
    </aside>
  )
}
