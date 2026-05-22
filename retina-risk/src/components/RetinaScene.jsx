import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import actualPredictions from '../data/actual_predictions.json'
import demoPredictions from '../data/demo_predictions.json'
import {
  createRetinaNode,
  createRetinaNodeConnections,
  getRetinaNodeColor,
  updateRetinaNode,
} from './Node'
import ReportOverlay from './ReportOverlay'
import SidePanel from './SidePanel'
import TrendChart from './TrendChart'

const datasets = {
  actual: {
    label: 'Actual',
    predictions: actualPredictions,
  },
  demo: {
    label: 'Demo',
    predictions: demoPredictions,
  },
}

function buildCohorts(predictions) {
  const defaultModel = predictions.metadata.default_model
  const burdenValues = predictions.cohorts.map((cohort) => cohort.disease_burden)
  const burdenMin = Math.min(...burdenValues)
  const burdenMax = Math.max(...burdenValues)
  const burdenRange = burdenMax - burdenMin
  const shouldNormalizeByRisk = burdenRange === 0
  const riskValues = predictions.cohorts.map((cohort) => cohort.models[defaultModel].risk_score)
  const riskMin = Math.min(...riskValues)
  const riskMax = Math.max(...riskValues)
  const riskRange = riskMax - riskMin || 1

  return predictions.cohorts.map((cohort) => {
    const model = cohort.models[defaultModel]
    const enrichedCohort = { ...cohort, model }

    return {
      ...enrichedCohort,
      color: `#${getRetinaNodeColor(enrichedCohort).getHexString()}`,
      depth_basis: shouldNormalizeByRisk ? 'risk score' : 'disease burden',
      normalized_disease_burden: shouldNormalizeByRisk
        ? (model.risk_score - riskMin) / riskRange
        : (cohort.disease_burden - burdenMin) / burdenRange,
    }
  })
}

export default function RetinaScene() {
  const mountRef = useRef(null)
  const [datasetKey, setDatasetKey] = useState('actual')
  const predictions = datasets[datasetKey].predictions
  const cohorts = useMemo(() => buildCohorts(predictions), [predictions])
  const [selectedCohort, setSelectedCohort] = useState(cohorts[0])
  const [chartCohort, setChartCohort] = useState(null)
  const [showReport, setShowReport] = useState(false)
  const activeCohort = cohorts.find((cohort) => cohort.id === selectedCohort?.id) ?? cohorts[0]

  function handleDatasetChange(nextDatasetKey) {
    const nextCohorts = buildCohorts(datasets[nextDatasetKey].predictions)

    setDatasetKey(nextDatasetKey)
    setSelectedCohort(nextCohorts[0])
    setChartCohort(null)
  }

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
        roughness: 0.18,
        metalness: 0.02,
        clearcoat: 0.6,
        transmission: 0.45,
        transparent: true,
        opacity: 0.2,
        depthWrite: false,
        side: THREE.DoubleSide,
      }),
    )
    shell.renderOrder = 1
    retina.add(shell)

    const retinaNodes = cohorts.map((cohort) => {
      const node = createRetinaNode(cohort)
      retina.add(node)
      return node
    })
    const nodeConnections = createRetinaNodeConnections(retinaNodes)
    retina.add(nodeConnections)

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
      nodeConnections.material.opacity = 0.18 + Math.sin(elapsedTime * 1.35) * 0.05
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
  }, [cohorts])

  return (
    <section className="retina-workspace">
      <div className="retina-scene" ref={mountRef} aria-label="Interactive 3D retina model" />
      <SidePanel
        cohorts={cohorts}
        datasetKey={datasetKey}
        datasetOptions={datasets}
        metadata={predictions.metadata}
        onDatasetChange={handleDatasetChange}
        onSelectCohort={setSelectedCohort}
        onShowChart={setChartCohort}
        onShowReport={() => setShowReport(true)}
        selectedCohort={activeCohort}
      />
      <TrendChart cohort={chartCohort} datasetKey={datasetKey} onClose={() => setChartCohort(null)} />
      {showReport && <ReportOverlay onClose={() => setShowReport(false)} />}
    </section>
  )
}
