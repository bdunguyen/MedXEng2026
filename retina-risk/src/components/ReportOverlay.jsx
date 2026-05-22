import { useEffect, useState } from 'react'
import { marked } from 'marked'

export default function ReportOverlay({ onClose }) {
  const [html, setHtml] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    fetch('/report.md')
      .then((r) => r.text())
      .then((md) => setHtml(marked.parse(md)))
      .catch(() => setError('Could not load report.'))
  }, [])

  return (
    <div className="report-overlay" role="dialog" aria-modal="true" aria-label="Project report">
      <div className="report-card">
        <header className="report-header">
          <h2>Project Report</h2>
          <button aria-label="Close report" onClick={onClose} type="button">
            x
          </button>
        </header>
        <div className="report-body">
          {error && <p>{error}</p>}
          {!error && !html && <p>Loading...</p>}
          {html && <div dangerouslySetInnerHTML={{ __html: html }} />}
        </div>
      </div>
    </div>
  )
}
