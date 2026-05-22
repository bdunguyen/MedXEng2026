import { createInsightForCohort } from '../server/geminiInsights.mjs'

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Use POST for /api/insights.' })
    return
  }

  try {
    const { cohortId, datasetKey } = req.body
    const insight = await createInsightForCohort(cohortId, datasetKey)
    res.status(200).json(insight)
  } catch (error) {
    res.status(error.statusCode || 500).json({ error: error.message || 'Unable to generate insight.' })
  }
}
