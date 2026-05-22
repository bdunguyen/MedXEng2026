import * as THREE from 'three'

const NODE_RADIUS_MAX = 1.86
const NODE_RADIUS_MIN = 0.08

function clamp(value, min = 0, max = 1) {
  return Math.min(Math.max(value, min), max)
}

function getNodeSeed(id) {
  return [...id].reduce((total, character) => total + character.charCodeAt(0), 0)
}

function getRiskColor(riskScore, seed = 0) {
  const low = new THREE.Color('#00d8ff')
  const mid = new THREE.Color('#ffd166')
  const high = new THREE.Color('#ff2e88')
  const color = new THREE.Color()
  const hueOffset = ((seed % 7) - 3) * 0.018

  if (riskScore < 0.5) {
    color.lerpColors(low, mid, riskScore / 0.5)
  } else {
    color.lerpColors(mid, high, (riskScore - 0.5) / 0.5)
  }

  color.offsetHSL(hueOffset, 0.08, riskScore > 0.7 ? 0.02 : -0.01)

  return color
}

export function isWorseningCohort(cohort) {
  const trend = cohort.model.trend

  if (trend.length < 2) {
    return false
  }

  return trend[trend.length - 1] > trend[0]
}

export function getRetinaNodePosition(cohort) {
  const burden = clamp(cohort.normalized_disease_burden ?? cohort.disease_burden)
  const source = new THREE.Vector3(cohort.position.x, cohort.position.y, cohort.position.z)
  const direction = source.lengthSq() > 0 ? source.normalize() : new THREE.Vector3(1, 0, 0)
  const centerDistance = NODE_RADIUS_MIN + (1 - burden) * (NODE_RADIUS_MAX - NODE_RADIUS_MIN)

  return direction.multiplyScalar(centerDistance)
}

export function createRetinaNodeConnections(nodes) {
  const positions = []
  const colors = []

  nodes.forEach((startNode, startIndex) => {
    nodes.slice(startIndex + 1).forEach((endNode) => {
      const startColor = getRiskColor(startNode.userData.riskScore, startNode.userData.seed)
      const endColor = getRiskColor(endNode.userData.riskScore, endNode.userData.seed)

      positions.push(
        startNode.position.x,
        startNode.position.y,
        startNode.position.z,
        endNode.position.x,
        endNode.position.y,
        endNode.position.z,
      )

      colors.push(
        startColor.r,
        startColor.g,
        startColor.b,
        endColor.r,
        endColor.g,
        endColor.b,
      )
    })
  })

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))

  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.28,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })

  const connections = new THREE.LineSegments(geometry, material)
  connections.name = 'retina-node-connections'
  connections.renderOrder = 2

  return connections
}

export function createRetinaNode(cohort) {
  const riskScore = clamp(cohort.model.risk_score)
  const seed = getNodeSeed(cohort.id)
  const riskColor = getRiskColor(riskScore, seed)
  const isWorsening = isWorseningCohort(cohort)
  const group = new THREE.Group()
  const radius = 0.055 + riskScore * 0.07
  const heat = 0.38 + riskScore * 1.35

  group.name = cohort.id
  group.userData = {
    cohortId: cohort.id,
    isRetinaNode: true,
    isWorsening,
    riskScore,
    pulseAmplitude: 0.07 + (seed % 5) * 0.025 + (isWorsening ? 0.08 : 0),
    pulsePhase: (seed % 17) * 0.37,
    pulseSpeed: 2.3 + (seed % 6) * 0.42 + (isWorsening ? 1.1 : 0),
    seed,
  }
  group.position.copy(getRetinaNodePosition(cohort))

  const core = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 32, 32),
    new THREE.MeshStandardMaterial({
      color: riskColor,
      emissive: riskColor,
      emissiveIntensity: heat,
      roughness: 0.2,
      metalness: 0.05,
    }),
  )
  core.name = cohort.id
  core.userData = group.userData
  group.add(core)

  const halo = new THREE.Mesh(
    new THREE.SphereGeometry(radius * (2.2 + riskScore), 32, 32),
    new THREE.MeshBasicMaterial({
      color: riskColor,
      transparent: true,
      opacity: 0.08 + riskScore * 0.12,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  )
  halo.name = `${cohort.id}-heat`
  halo.userData = group.userData
  group.add(halo)

  group.lookAt(0, 0, 0)
  group.userData.core = core
  group.userData.halo = halo

  return group
}

export function updateRetinaNode(node, elapsedTime) {
  const { halo, isWorsening, pulseAmplitude, pulsePhase, pulseSpeed, riskScore } = node.userData
  const wave = Math.sin(elapsedTime * pulseSpeed + pulsePhase)
  const pulse = 1 + wave * pulseAmplitude
  const heatPulse = 1 + Math.sin(elapsedTime * (pulseSpeed * 0.72) + pulsePhase * 1.6) * 0.1 * riskScore

  node.scale.setScalar(pulse)

  if (halo) {
    halo.scale.setScalar(heatPulse)
    halo.material.opacity =
      0.08 + riskScore * 0.12 + (isWorsening ? Math.max(pulse - 1, 0) * 0.36 : 0)
  }
}
