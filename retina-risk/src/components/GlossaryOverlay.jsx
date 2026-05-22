const terms = [
  {
    term: 'AUROC',
    def: 'Area Under the ROC Curve. Measures how well the model ranks high-risk patients above low-risk ones. 0.50 = coin flip, 1.00 = perfect. Our holdout AUROC of 0.718 means the model correctly ranks the higher-risk patient 71.8% of the time when comparing any two patients.',
  },
  {
    term: 'AUPRC',
    def: 'Area Under the Precision-Recall Curve. Like AUROC but more sensitive when events are rare. A random model scores equal to the event rate (e.g. 0.21 at 21% prevalence). Our model scores 0.357 — 1.7× better than random.',
  },
  {
    term: 'Brier Score',
    def: 'Measures how far predicted probabilities are from actual outcomes. 0 = perfect, 0.25 = uninformative (equivalent to always predicting 50%). Lower is better.',
  },
  {
    term: 'Calibration',
    def: 'Whether a model\'s probability scores are accurate in absolute terms. A well-calibrated model where it says "10% risk" means roughly 1 in 10 patients actually have an event. Our model was calibrated on ~4% prevalence data, so scores underestimate the true 20.9% holdout event rate.',
  },
  {
    term: 'Sensitivity (Recall)',
    def: 'Of all patients who actually had an event, what fraction did the model flag? At our threshold of 0.589, sensitivity is 30% — the model catches 3 in 10 true events.',
  },
  {
    term: 'Specificity',
    def: 'Of all patients who did not have an event, what fraction did the model correctly leave unflagged? At our threshold: 86%.',
  },
  {
    term: 'PPV (Precision)',
    def: 'Of all patients the model flags as high-risk, what fraction actually have an event? At our threshold: 9%. In practice: contact ~11 flagged patients to reach 1 who would have had an acute event.',
  },
  {
    term: 'Lift',
    def: 'How much better the model is than random selection at a given depth. A 2× lift in the top decile means the flagged group has twice the event rate of a randomly chosen group the same size.',
  },
  {
    term: 'Cohort',
    def: 'A patient subgroup defined by a shared condition (e.g. Glaucoma only, Diabetic only, Multi-condition). Each cohort is evaluated separately to assess model performance and fairness.',
  },
  {
    term: 'Disease Burden',
    def: 'A composite severity index used to position each cohort node in the 3D visualization. Higher burden = closer to center. Based on clinical complexity: multi-condition patients score highest, glaucoma-only patients score lowest.',
  },
  {
    term: 'ADI (Area Deprivation Index)',
    def: 'A neighborhood-level measure of socioeconomic disadvantage. Higher ADI = more deprived. Patients in the most deprived quintile (Q5) had a 21% event rate vs 13.5% in the least deprived — a 1.6× gap.',
  },
  {
    term: 'Isotonic Calibration',
    def: 'A post-processing step applied to the model\'s raw probability scores to make them more accurate in absolute terms. We fit an isotonic regression on the validation set to adjust the RF\'s output probabilities.',
  },
]

export default function GlossaryOverlay({ onClose }) {
  return (
    <div className="report-overlay" role="dialog" aria-modal="true" aria-label="Glossary of terms">
      <div className="report-card">
        <header className="report-header">
          <h2>Glossary</h2>
          <button aria-label="Close glossary" onClick={onClose} type="button">
            x
          </button>
        </header>
        <div className="report-body glossary-body">
          {terms.map(({ term, def }) => (
            <div className="glossary-entry" key={term}>
              <strong>{term}</strong>
              <p>{def}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
