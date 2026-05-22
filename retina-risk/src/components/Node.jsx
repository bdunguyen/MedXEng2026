import * as THREE from 'three'

const RETINA_RADIUS = 2.07
const CENTER_PULL_MAX = 1.45
const CENTER_PULL_MIN = 0.12

function clamp(value, min = 0, max = 1) {
  return Math.min(Math.max(value, min), max)
}

function getRiskColor(riskScore) {
  const low = new THREE.Color('#22c55e')
  const mid = new THREE.Color('#f59e0b')
  const high = new THREE.Color('#ef4444')
  const color = new THREE.Color()

  if (riskScore < 0.5) {
    return color.lerpColors(low, mid, riskScore / 0.5)
  }

  return color.lerpColors(mid, high, (riskScore - 0.5) / 0.5)
}

export function isWorseningCohort(cohort) {
  const trend = cohort.model.trend

  if (trend.length < 2) {
    return false
  }

  return trend[trend.length - 1] > trend[0]
}

export function getRetinaNodePosition(cohort) {
  const burden = clamp(cohort.disease_burden)
  const source = new THREE.Vector2(cohort.position.x, cohort.position.y)
  const direction = source.lengthSq() > 0 ? source.normalize() : new THREE.Vector2(1, 0)
  const centerDistance = CENTER_PULL_MIN + (1 - burden) * CENTER_PULL_MAX
  const x = direction.x * centerDistance
  const y = direction.y * centerDistance
  const z = Math.sqrt(Math.max(RETINA_RADIUS ** 2 - x ** 2 - y ** 2, 0))

  return new THREE.Vector3(x, y, z)
}

export function createRetinaNode(cohort) {
  const riskScore = clamp(cohort.model.risk_score)
  const riskColor = getRiskColor(riskScore)
  const isWorsening = isWorseningCohort(cohort)
  const group = new THREE.Group()
  const radius = 0.055 + riskScore * 0.07
  const heat = 0.28 + riskScore * 1.2

  group.name = cohort.id
  group.userData = {
    cohortId: cohort.id,
    isRetinaNode: true,
    isWorsening,
    riskScore,
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
      opacity: 0.1 + riskScore * 0.16,
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
  const { halo, isWorsening, riskScore } = node.userData
  const pulse = isWorsening ? 1 + Math.sin(elapsedTime * 4.5) * 0.16 : 1
  const heatPulse = 1 + Math.sin(elapsedTime * 3.2) * 0.08 * riskScore

  node.scale.setScalar(pulse)

  if (halo) {
    halo.scale.setScalar(heatPulse)
    halo.material.opacity = 0.1 + riskScore * 0.16 + (isWorsening ? (pulse - 1) * 0.28 : 0)
  }
}
