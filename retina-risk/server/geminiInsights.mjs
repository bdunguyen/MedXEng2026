import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { GoogleGenAI, Type } from '@google/genai'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
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
    source: 'local-fallback',
    summary: `${cohort.name} has a ${formatPercent(cohort.model.risk_score)} ${cohort.model.risk_level} model risk score across ${cohort.population} patients.`,
    riskDrivers: features.length ? features : ['No ranked model features were provided for this model.'],
    trendInterpretation: `The six-point trend is ${trendDirection}.`,
    chartCallout: `Use the chart to compare current risk, trend direction, and cohort size for ${metadata.default_model}.`,
    caveats: [
      'Generated locally because GEMINI_API_KEY is not configured.',
      'This is model interpretation support, not a diagnosis or treatment recommendation.',
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

  if (!process.env.GEMINI_API_KEY) {
    console.log('[Gemini] No GEMINI_API_KEY in environment, using fallback')
    return makeFallbackInsight(selectedCohort, predictions.metadata)
  }

  try {
    console.log(`[Gemini] Initializing with model: ${process.env.GEMINI_MODEL || 'gemini-2.0-flash'}`)
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY })
    const response = await ai.models.generateContent({
      model: process.env.GEMINI_MODEL || 'gemini-2.0-flash',
      contents: buildPrompt(selectedCohort, predictions.metadata),
      config: {
        responseMimeType: 'application/json',
        responseSchema: insightSchema,
        systemInstruction:
          'You summarize machine learning prediction data for an ophthalmology dashboard. Keep statements concise, cautious, and non-diagnostic.',
      },
    })
    console.log('[Gemini] Successfully received response')
    const insight = validateInsight(JSON.parse(response.text))

    return {
      source: 'gemini',
      ...insight,
    }
  } catch (error) {
    console.error('[Gemini] API call failed:', error.message)
    console.error('[Gemini] Full error:', error)
    const fallback = makeFallbackInsight(selectedCohort, predictions.metadata)

    return {
      ...fallback,
      caveats: [
        'Gemini was configured but the request failed, so this insight used the local fallback.',
        error.message?.includes('quota') ? 'Gemini reported a quota or rate-limit issue.' : `Gemini did not return a usable response: ${error.message}`,
        ...fallback.caveats.slice(1),
      ],
    }
  }
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
