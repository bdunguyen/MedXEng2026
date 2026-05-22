import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import predictions from '../data/predictions.json'
import { createRetinaNode, updateRetinaNode } from './Node'
import TrendChart from './TrendChart'

const defaultModel = predictions.metadata.default_model

const riskColors = {
  high: '#ef4444',
  moderate: '#f59e0b',
  low: '#22c55e',
}

const cohorts = predictions.cohorts.map((cohort) => {
  const model = cohort.models[defaultModel]

  return {
    ...cohort,
    model,
    color: riskColors[model.risk_level] ?? '#38bdf8',
  }
})

function makeVessel(points, color = '#d94141') {
  const curve = new THREE.CatmullRomCurve3(points)
  const geometry = new THREE.TubeGeometry(curve, 48, 0.018, 8, false)
  const material = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.42,
    metalness: 0.04,
    emissive: color,
    emissiveIntensity: 0.16,
  })

  return new THREE.Mesh(geometry, material)
}

function toSurfacePoint(x, y, z = 2.05) {
  return new THREE.Vector3(x, y, z).normalize().multiplyScalar(2.04)
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`
}

export default function RetinaScene() {
  const mountRef = useRef(null)
  const [selectedCohort, setSelectedCohort] = useState(cohorts[0])
  const [chartCohort, setChartCohort] = useState(null)

  useEffect(() => {
    const mount = mountRef.current
    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#071014')

    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100)
    camera.position.set(0, 0.4, 6)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.setSize(mount.clientWidth, mount.clientHeight)
    mount.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.autoRotate = true
    controls.autoRotateSpeed = 0.45
    controls.minDistance = 3.6
    controls.maxDistance = 8

    const retina = new THREE.Group()
    scene.add(retina)

    scene.add(new THREE.AmbientLight('#adc5ff', 0.85))

    const keyLight = new THREE.DirectionalLight('#ffffff', 2.8)
    keyLight.position.set(2.4, 3, 4.5)
    scene.add(keyLight)

    const rimLight = new THREE.PointLight('#3dd6ff', 28, 14)
    rimLight.position.set(-3.5, -2.2, 3.5)
    scene.add(rimLight)

    const shell = new THREE.Mesh(
      new THREE.SphereGeometry(2, 96, 96),
      new THREE.MeshPhysicalMaterial({
        color: '#9be7dc',
        roughness: 0.52,
        metalness: 0.04,
        clearcoat: 0.35,
        transmission: 0.08,
        transparent: true,
        opacity: 0.86,
      }),
    )
    retina.add(shell)

    const surface = new THREE.Mesh(
      new THREE.SphereGeometry(2.012, 96, 96, 0, Math.PI * 2, 0, Math.PI),
      new THREE.MeshStandardMaterial({
        color: '#132d2f',
        emissive: '#123c3e',
        emissiveIntensity: 0.28,
        roughness: 0.7,
        transparent: true,
        opacity: 0.62,
      }),
    )
    retina.add(surface)

    const macula = new THREE.Mesh(
      new THREE.CircleGeometry(0.42, 64),
      new THREE.MeshBasicMaterial({
        color: '#f7b733',
        transparent: true,
        opacity: 0.52,
        side: THREE.DoubleSide,
      }),
    )
    macula.position.set(0, 0, 2.035)
    retina.add(macula)

    const opticDisc = new THREE.Mesh(
      new THREE.CircleGeometry(0.28, 64),
      new THREE.MeshBasicMaterial({
        color: '#f8fafc',
        transparent: true,
        opacity: 0.68,
        side: THREE.DoubleSide,
      }),
    )
    opticDisc.position.copy(toSurfacePoint(0.8, -0.35))
    opticDisc.lookAt(0, 0, 0)
    retina.add(opticDisc)

    const vesselGroup = new THREE.Group()
    vesselGroup.add(
      makeVessel([
        toSurfacePoint(0.72, -0.32),
        toSurfacePoint(0.32, 0.08),
        toSurfacePoint(-0.34, 0.42),
        toSurfacePoint(-1.12, 0.72),
      ]),
      makeVessel([
        toSurfacePoint(0.76, -0.3),
        toSurfacePoint(0.22, -0.46),
        toSurfacePoint(-0.42, -0.62),
        toSurfacePoint(-1.08, -0.76),
      ]),
      makeVessel([
        toSurfacePoint(0.76, -0.28),
        toSurfacePoint(1.04, 0.2),
        toSurfacePoint(1.36, 0.62),
      ], '#ff6b6b'),
    )
    retina.add(vesselGroup)

    const retinaNodes = cohorts.map((cohort) => {
      const node = createRetinaNode(cohort)
      retina.add(node)
      return node
    })
    const markerMeshes = retinaNodes.flatMap((node) => node.children)

    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()

    function resize() {
      const { clientWidth, clientHeight } = mount
      camera.aspect = clientWidth / clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(clientWidth, clientHeight)
    }

    function setPointer(event) {
      const rect = renderer.domElement.getBoundingClientRect()
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
    }

    function handlePointerMove(event) {
      setPointer(event)
      raycaster.setFromCamera(pointer, camera)
      const intersects = raycaster.intersectObjects(markerMeshes)
      renderer.domElement.style.cursor = intersects.length ? 'pointer' : 'grab'
    }

    function handlePointerDown(event) {
      controls.autoRotate = false
      setPointer(event)
      raycaster.setFromCamera(pointer, camera)
      const [hit] = raycaster.intersectObjects(markerMeshes)

      if (hit) {
        const nextCohort = cohorts.find((cohort) => cohort.id === hit.object.userData.cohortId)
        setSelectedCohort(nextCohort)
        setChartCohort(nextCohort)
      }
    }

    const clock = new THREE.Clock()
    let frameId
    function animate() {
      frameId = requestAnimationFrame(animate)
      const elapsedTime = clock.getElapsedTime()
      retinaNodes.forEach((node) => updateRetinaNode(node, elapsedTime))
      controls.update()
      renderer.render(scene, camera)
    }

    resize()
    animate()

    window.addEventListener('resize', resize)
    renderer.domElement.addEventListener('pointermove', handlePointerMove)
    renderer.domElement.addEventListener('pointerdown', handlePointerDown)

    return () => {
      cancelAnimationFrame(frameId)
      window.removeEventListener('resize', resize)
      renderer.domElement.removeEventListener('pointermove', handlePointerMove)
      renderer.domElement.removeEventListener('pointerdown', handlePointerDown)
      controls.dispose()
      renderer.dispose()
      mount.removeChild(renderer.domElement)
      scene.traverse((object) => {
        object.geometry?.dispose()
        if (Array.isArray(object.material)) {
          object.material.forEach((material) => material.dispose())
        } else {
          object.material?.dispose()
        }
      })
    }
  }, [])

  return (
    <section className="retina-workspace">
      <div className="retina-scene" ref={mountRef} aria-label="Interactive 3D retina model" />
      <aside className="retina-panel">
        <div className="project-signature" aria-label="MedXEng2026 Britney and Leo">
          <span>MedXEng2026</span>
          <span>Britney &amp; Leo</span>
        </div>
        <div className="prediction-meta">
          <p>{predictions.metadata.project}</p>
          <p>{predictions.metadata.prediction_target}</p>
        </div>
        <div className="finding-readout cohort-readout">
          <span style={{ backgroundColor: selectedCohort.color }} />
          <div>
            <h2>{selectedCohort.name}</h2>
            <p>
              {formatPercent(selectedCohort.model.risk_score)} {selectedCohort.model.risk_level}{' '}
              risk · {selectedCohort.population} patients
            </p>
          </div>
        </div>
        <div className="risk-meter" aria-label={`${selectedCohort.name} risk score`}>
          <div style={{ width: formatPercent(selectedCohort.model.risk_score) }} />
        </div>
        <div className="feature-list">
          <h3>Top features</h3>
          {selectedCohort.model.top_features.map((item) => (
            <div className="feature-row" key={item.feature}>
              <span>{item.feature}</span>
              <strong>{formatPercent(item.importance)}</strong>
            </div>
          ))}
        </div>
        <div className="cohort-list" aria-label="Prediction cohorts">
          {cohorts.map((cohort) => (
            <button
              className={cohort.id === selectedCohort.id ? 'active' : ''}
              key={cohort.id}
              onClick={() => setSelectedCohort(cohort)}
              type="button"
            >
              <span style={{ backgroundColor: cohort.color }} />
              {cohort.name}
            </button>
          ))}
        </div>
      </aside>
      <TrendChart cohort={chartCohort} onClose={() => setChartCohort(null)} />
    </section>
  )
}
