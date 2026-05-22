import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { GoogleGenAI, Type } from '@google/genai'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const insightCache = new Map()

const predictionPaths = {
  actual: path.resolve(__dirname, '../src/data/actual_predictions.json'),
  demo: path.resolve(__dirname, '../src/data/demo_predictions.json'),
}

const insightSchema = {
  type: Type.OBJECT,
  properties: {
    summary: { type: Type.STRING },
    riskDrivers: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
    },
    trendInterpretation: { type: Type.STRING },
    chartCallout: { type: Type.STRING },
    caveats: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
    },
  },
  propertyOrdering: ['summary', 'riskDrivers', 'trendInterpretation', 'chartCallout', 'caveats'],
  required: ['summary', 'riskDrivers', 'trendInterpretation', 'chartCallout', 'caveats'],
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`
}

function getTrendDirection(trend) {
  if (!Array.isArray(trend) || trend.length < 2) {
    return 'limited trend data'
  }

  const first = trend[0]
  const last = trend[trend.length - 1]

  if (last > first) {
    return `worsening from ${formatPercent(first)} to ${formatPercent(last)}`
  }

  if (last < first) {
    return `improving from ${formatPercent(first)} to ${formatPercent(last)}`
  }

  return `flat at ${formatPercent(last)}`
}

function makeFallbackInsight(cohort, metadata) {
  const trendDirection = getTrendDirection(cohort.model.trend)
  const features = cohort.model.top_features
    .filter((feature) => feature.feature !== 'N/A')
    .map((feature) => `${feature.feature} (${formatPercent(feature.importance)})`)

  return {
    source: 'summary',
    summary: `${cohort.name} cohort: ${formatPercent(cohort.model.risk_score)} predicted 12-month admission risk across ${cohort.population} patients. Risk level is ${cohort.model.risk_level} relative to the full cohort.`,
    riskDrivers: features.length ? features : ['Feature importance not available for this model type.'],
    trendInterpretation: `Six-point trend is ${trendDirection}. Each point represents a rolling risk estimate across the observation window.`,
    chartCallout: `Higher scores indicate greater model-predicted probability of ED visit or inpatient admission within 12 months. Compare across cohorts using the node positions in the 3D view.`,
    caveats: [
      'This is decision-support information, not a clinical diagnosis or treatment recommendation.',
      'Risk scores should be reviewed alongside full patient context before any outreach decision.',
    ],
  }
}

function buildPrompt(cohort, metadata) {
  return `
Analyze this ophthalmology prediction cohort for a visualization tooltip.

Return concise decision-support language. Do not diagnose, prescribe, or imply certainty.
Use "model-associated", "may indicate", or "should be reviewed" phrasing when appropriate.

Metadata:
${JSON.stringify(metadata, null, 2)}

Selected cohort:
${JSON.stringify(cohort, null, 2)}
`.trim()
}

function validateInsight(value) {
  if (!value || typeof value !== 'object') {
    throw new Error('Insight response was not an object.')
  }

  const requiredStrings = ['summary', 'trendInterpretation', 'chartCallout']
  requiredStrings.forEach((key) => {
    if (typeof value[key] !== 'string') {
      throw new Error(`Insight response missing string field: ${key}`)
    }
  })

  if (!Array.isArray(value.riskDrivers) || !Array.isArray(value.caveats)) {
    throw new Error('Insight response missing list fields.')
  }

  return value
}

async function loadPredictions(datasetKey = 'actual') {
  const predictionsPath = predictionPaths[datasetKey]

  if (!predictionsPath) {
    const error = new Error(`Unknown dataset key: ${datasetKey}`)
    error.statusCode = 400
    throw error
  }

  return JSON.parse(await readFile(predictionsPath, 'utf8'))
}

async function getRequestBody(req) {
  const chunks = []

  for await (const chunk of req) {
    chunks.push(chunk)
  }

  if (!chunks.length) {
    return {}
  }

  return JSON.parse(Buffer.concat(chunks).toString('utf8'))
}

export async function createInsightForCohort(cohortId, datasetKey = 'actual') {
  const cacheKey = `${datasetKey}:${cohortId}`
  if (insightCache.has(cacheKey)) {
    return insightCache.get(cacheKey)
  }

  const predictions = await loadPredictions(datasetKey)
  const defaultModel = predictions.metadata.default_model
  const cohort = predictions.cohorts.find((item) => item.id === cohortId)

  if (!cohort) {
    const error = new Error(`Unknown cohort id: ${cohortId}`)
    error.statusCode = 404
    throw error
  }

  const selectedCohort = {
    ...cohort,
    model: cohort.models[defaultModel],
  }

  const fallback = makeFallbackInsight(selectedCohort, predictions.metadata)
  insightCache.set(cacheKey, fallback)
  return fallback
}

export async function handleInsightsRequest(req, res) {
  try {
    if (req.method !== 'POST') {
      res.statusCode = 405
      res.setHeader('content-type', 'application/json')
      res.end(JSON.stringify({ error: 'Use POST for /api/insights.' }))
      return
    }

    const body = await getRequestBody(req)
    const insight = await createInsightForCohort(body.cohortId, body.datasetKey)

    res.statusCode = 200
    res.setHeader('content-type', 'application/json')
    res.end(JSON.stringify(insight))
  } catch (error) {
    res.statusCode = error.statusCode || 500
    res.setHeader('content-type', 'application/json')
    res.end(JSON.stringify({ error: error.message || 'Unable to generate insight.' }))
  }
}
