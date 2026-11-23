import { useEffect, useRef, useState } from 'react'

import PageNavigation from '../components/PageNavigation'

import {
  getDocuments,
  uploadDocument,
  type DocumentInfo,
  type DocumentVersion,
} from '../services/api'

import { getSavedLanguage } from '../i18n/language'
import { documentTranslations } from '../i18n/documentTranslations'

import './DocumentsPage.css'

// Maximum upload size: 10 MB (matches FastAPI)
const MAX_FILE_SIZE = 10 * 1024 * 1024

// Special values used only by the frontend
const OTHER_OPTION = '__other__'

function formatDate(
  dateString: string | null,
  unspecifiedText: string
): string {
  if (!dateString) {
    return unspecifiedText
  }

  // Handle date-only values without timezone conversion
  const dateOnlyMatch = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dateString)

  if (dateOnlyMatch) {
    const [, year, month, day] = dateOnlyMatch

    return new Date(
      Number(year),
      Number(month) - 1,
      Number(day)
    ).toLocaleDateString()
  }

  // Handle timestamps normally
  const date = new Date(dateString)

  if (Number.isNaN(date.getTime())) {
    return dateString
  }

  return date.toLocaleDateString()
}

function formatFileSize(size: number): string {
  return `${(size / (1024 * 1024)).toFixed(2)} MB`
}

function DocumentsPage() {
  // Load the user's saved language preference
  const language = getSavedLanguage()
  const t = documentTranslations[language]

  // Reference to the actual HTML file input
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Document list
  const [documents, setDocuments] = useState<DocumentInfo[]>([])

  // Upload form
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [documentName, setDocumentName] = useState('')
  const [documentVersion, setDocumentVersion] = useState('1.0')
  const [effectiveDate, setEffectiveDate] = useState('')

  // Authority dropdown and custom value
  const [authority, setAuthority] = useState('')
  const [customAuthority, setCustomAuthority] = useState('')

  // Department dropdown and custom value
  const [ownerDepartment, setOwnerDepartment] = useState('')
  const [customDepartment, setCustomDepartment] = useState('')

  // Loading and feedback
  const [isLoading, setIsLoading] = useState(true)
  const [isUploading, setIsUploading] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')
  const [successMessage, setSuccessMessage] = useState('')

  // Load documents from FastAPI
  const loadDocuments = async () => {
    try {
      setIsLoading(true)
      setErrorMessage('')

      const result = await getDocuments()

      setDocuments(result.documents)

    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : t.loadError
      )

    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadDocuments()
  }, [])

  // Open the native file selection dialog
  const handleChooseFile = () => {
    fileInputRef.current?.click()
  }

  // Handle file selection
  const handleFileChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0] ?? null

    setErrorMessage('')
    setSuccessMessage('')

    if (!file) {
      setSelectedFile(null)
      return
    }

    const extension = file.name.split('.').pop()?.toLowerCase()

    if (extension !== 'pdf' && extension !== 'txt') {
      setErrorMessage(t.invalidFile)
      event.target.value = ''
      setSelectedFile(null)
      return
    }

    if (file.size === 0) {
      setErrorMessage(t.invalidFile)
      event.target.value = ''
      setSelectedFile(null)
      return
    }

    if (file.size > MAX_FILE_SIZE) {
      setErrorMessage(t.fileTooLarge)
      event.target.value = ''
      setSelectedFile(null)
      return
    }

    setSelectedFile(file)

    // Use the filename as the default document name
    if (!documentName.trim()) {
      setDocumentName(
        file.name.replace(/\.[^/.]+$/, '')
      )
    }
  }

  // Upload document
  const handleUpload = async (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault()

    if (!selectedFile || isUploading) {
      return
    }

    // Resolve dropdown selections into backend metadata values
    const authorityValue =
      authority === OTHER_OPTION
        ? customAuthority.trim()
        : authority.trim()

    const departmentValue =
      ownerDepartment === OTHER_OPTION
        ? customDepartment.trim()
        : ownerDepartment.trim()

    // Validate custom values when Other is selected
    if (authority === OTHER_OPTION && !authorityValue) {
      setErrorMessage(t.customValueRequired)
      return
    }

    if (ownerDepartment === OTHER_OPTION && !departmentValue) {
      setErrorMessage(t.customValueRequired)
      return
    }

    try {
      setIsUploading(true)
      setErrorMessage('')
      setSuccessMessage('')

      await uploadDocument({
        file: selectedFile,
        document_name: documentName.trim() || undefined,
        document_version: documentVersion.trim() || undefined,
        effective_date: effectiveDate || undefined,
        authority: authorityValue || undefined,
        owner_department: departmentValue || undefined,
      })

      setSuccessMessage(t.uploadSuccess)

      // Clear form after successful upload
      setSelectedFile(null)
      setDocumentName('')
      setDocumentVersion('1.0')
      setEffectiveDate('')

      setAuthority('')
      setCustomAuthority('')

      setOwnerDepartment('')
      setCustomDepartment('')

      // Reset the actual HTML file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }

      // Refresh document list
      await loadDocuments()

    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : t.uploadError
      )

    } finally {
      setIsUploading(false)
    }
  }

  // Render one document version
  const renderVersion = (version: DocumentVersion) => {
    return (
      <div
        key={version.version_id}
        className="document-version"
      >
        <div className="version-header">

          <strong>
            {t.version} {version.document_version}
          </strong>

          <span
            className={`version-status ${version.version_status}`}
          >
            {version.version_status === 'current'
              ? t.current
              : version.version_status}
          </span>

        </div>

        <div className="version-details">

          <p>
            <strong>{t.effectiveDate}:</strong>{' '}
            {formatDate(version.effective_date, t.unspecified)}
          </p>

          <p>
            <strong>{t.uploaded}:</strong>{' '}
            {formatDate(version.upload_date, t.unspecified)}
          </p>

          <p>
            <strong>{t.authority}:</strong>{' '}
            {version.authority || t.unspecified}
          </p>

          <p>
            <strong>{t.department}:</strong>{' '}
            {version.owner_department || t.unspecified}
          </p>

          <p>
            <strong>{t.chunks}:</strong>{' '}
            {version.chunks}
          </p>

          {version.source && (
            <p>
              <strong>{t.filename}:</strong>{' '}
              {version.source}
            </p>
          )}

        </div>
      </div>
    )
  }

  // ==========================================
  // Render UI
  // ==========================================

  return (
    <>
      {/* Shared Navigation */}
      <PageNavigation />

      {/* Documents Page */}
      <div className="documents-container">

        {/* Page Header */}
        <header className="documents-header">

          <div>
            <h1>{t.title}</h1>

            <p>
              {t.description}
            </p>
          </div>

          <button
            type="button"
            className="refresh-button"
            onClick={loadDocuments}
            disabled={isLoading}
          >
            {isLoading ? t.loading : t.refresh}
          </button>

        </header>

        {/* Feedback */}
        {errorMessage && (
          <div className="documents-error">
            {errorMessage}
          </div>
        )}

        {successMessage && (
          <div className="documents-success">
            {successMessage}
          </div>
        )}

        {/* Upload Form */}
        <section className="upload-section">

          <h2>{t.uploadTitle}</h2>

          <p className="section-description">
            {t.uploadDescription}
          </p>

          <form onSubmit={handleUpload}>

            <div className="form-group">

              <label htmlFor="document-file">
                {t.selectFile} *
              </label>

              {/* Hidden native file input */}
              <input
                ref={fileInputRef}
                id="document-file"
                type="file"
                accept=".pdf,.txt"
                onChange={handleFileChange}
                disabled={isUploading}
                required
                className="hidden-file-input"
                tabIndex={-1}
                aria-hidden="true"
              />

              {/* Custom translated file picker */}
              <div className="custom-file-picker">

                <button
                  type="button"
                  className="choose-file-button"
                  onClick={handleChooseFile}
                  disabled={isUploading}
                >
                  {t.chooseFile}
                </button>

                <span className="selected-file-name">
                  {selectedFile
                    ? `${selectedFile.name} (${formatFileSize(selectedFile.size)})`
                    : t.noFileSelected}
                </span>

              </div>

            </div>

            <div className="form-grid">

              <div className="form-group">

                <label htmlFor="document-name">
                  {t.documentName}
                </label>

                <input
                  id="document-name"
                  type="text"
                  value={documentName}
                  onChange={(event) =>
                    setDocumentName(event.target.value)
                  }
                  placeholder={t.documentName}
                  disabled={isUploading}
                />

              </div>

              <div className="form-group">

                <label htmlFor="document-version">
                  {t.version}
                </label>

                <input
                  id="document-version"
                  type="text"
                  value={documentVersion}
                  onChange={(event) =>
                    setDocumentVersion(event.target.value)
                  }
                  placeholder="1.0"
                  disabled={isUploading}
                />

              </div>

              <div className="form-group">

                <label htmlFor="effective-date">
                  {t.effectiveDate}
                </label>

                <input
                  id="effective-date"
                  type="date"
                  value={effectiveDate}
                  onChange={(event) =>
                    setEffectiveDate(event.target.value)
                  }
                  disabled={isUploading}
                />

              </div>

              {/* Authority dropdown */}
              <div className="form-group">

                <label htmlFor="authority">
                  {t.authority}
                </label>

                <select
                  id="authority"
                  value={authority}
                  onChange={(event) => {
                    setAuthority(event.target.value)
                    setCustomAuthority('')
                  }}
                  disabled={isUploading}
                >
                  <option value="">
                    {t.selectAuthority}
                  </option>

                  <option value="Official Policy">
                    {t.officialPolicy}
                  </option>

                  <option value="Employee Handbook">
                    {t.employeeHandbook}
                  </option>

                  <option value="Department Policy">
                    {t.departmentPolicy}
                  </option>

                  <option value="Regulatory Guidance">
                    {t.regulatoryGuidance}
                  </option>

                  <option value={OTHER_OPTION}>
                    {t.other}
                  </option>

                </select>

                {authority === OTHER_OPTION && (
                  <input
                    type="text"
                    value={customAuthority}
                    onChange={(event) =>
                      setCustomAuthority(event.target.value)
                    }
                    placeholder={t.enterCustomValue}
                    disabled={isUploading}
                    required
                  />
                )}

              </div>

              {/* Department dropdown */}
              <div className="form-group">

                <label htmlFor="owner-department">
                  {t.ownerDepartment}
                </label>

                <select
                  id="owner-department"
                  value={ownerDepartment}
                  onChange={(event) => {
                    setOwnerDepartment(event.target.value)
                    setCustomDepartment('')
                  }}
                  disabled={isUploading}
                >
                  <option value="">
                    {t.selectDepartment}
                  </option>

                  <option value="Human Resources">
                    {t.humanResources}
                  </option>

                  <option value="Information Technology">
                    {t.informationTechnology}
                  </option>

                  <option value="Legal">
                    {t.legal}
                  </option>

                  <option value="Finance">
                    {t.finance}
                  </option>

                  <option value="Compliance">
                    {t.compliance}
                  </option>

                  <option value="Engineering">
                    {t.engineering}
                  </option>

                  <option value="Operations">
                    {t.operations}
                  </option>

                  <option value="Sales">
                    {t.sales}
                  </option>

                  <option value="Customer Support">
                    {t.customerSupport}
                  </option>

                  <option value={OTHER_OPTION}>
                    {t.other}
                  </option>

                </select>

                {ownerDepartment === OTHER_OPTION && (
                  <input
                    type="text"
                    value={customDepartment}
                    onChange={(event) =>
                      setCustomDepartment(event.target.value)
                    }
                    placeholder={t.enterCustomValue}
                    disabled={isUploading}
                    required
                  />
                )}

              </div>

            </div>

            <button
              type="submit"
              className="upload-button"
              disabled={!selectedFile || isUploading}
            >
              {isUploading ? t.uploading : t.upload}
            </button>

          </form>

        </section>

        {/* Uploaded Documents */}
        <section className="document-list-section">

          <div className="document-list-header">

            <h2>{t.uploadedDocuments}</h2>

            <span>
              {documents.length} {t.documents}
            </span>

          </div>

          {isLoading && (
            <p>{t.loading}</p>
          )}

          {!isLoading && documents.length === 0 && (
            <div className="empty-documents">
              {t.noDocuments}
            </div>
          )}

          {!isLoading && documents.length > 0 && (

            <div className="document-list">

              {documents.map((item) => (

                <details
                  key={`${item.store_type}-${item.document_id}`}
                  className="document-card"
                  open
                >

                  <summary className="document-summary">

                    <div>
                      <h3>{item.document_name}</h3>

                      <p>
                        {item.is_shared ? t.shared : t.private}
                        {' · '}
                        {item.versions.length} {t.versions}
                      </p>
                    </div>

                    <span className="expand-indicator">
                      ▼
                    </span>

                  </summary>

                  <div className="document-versions">

                    {item.versions.map(renderVersion)}

                  </div>

                </details>

              ))}

            </div>

          )}

        </section>

      </div>
    </>
  )
}

export default DocumentsPage