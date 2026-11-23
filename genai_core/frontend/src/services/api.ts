// FastAPI backend URL from environment configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL

if (!API_BASE_URL) {
  throw new Error('VITE_API_BASE_URL is not configured')
}

// ==================================================
// Request / Response Interfaces
// ==================================================

// Supported languages
export type SupportedLanguage =
  | 'en'
  | 'zh-CN'
  | 'zh-TW'
  | 'de'
  | 'fr'
  | 'ru'


// Login request structure
export interface LoginRequest {
  email: string
  password: string
}

// Login response structure
export interface LoginResponse {
  status: string
  access_token: string
  token_type: string
}

// Chat session structure
export interface SessionInfo {
  title: string
  persona: string
  last_activity: string | null
}

// Message structure
export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

// Create session response
export interface CreateSessionResponse {
  session_id: string
}

// Send message response
export interface SendMessageResponse {
  reply: string
}

// Translation response
export interface TranslateResponse {
  original_text: string
  translated_text: string
  target_language: SupportedLanguage
}


// ==================================================
// Document Interfaces
// ==================================================

// Document upload request
export interface UploadDocumentRequest {
  file: File
  document_name?: string
  document_version?: string
  effective_date?: string
  authority?: string
  owner_department?: string
}

// Document upload response
export interface UploadDocumentResponse {
  status: string
  filename: string
  size: number

  metadata: {
    document_name: string
    document_version: string | null
    effective_date: string | null
    authority: string | null
    upload_date: string
    owner_department: string | null
  }

  message: string
}

// Individual document version
export interface DocumentVersion {
  version_id: string
  source: string | null
  document_name: string | null
  document_version: string
  version_status: string
  is_current: boolean
  effective_date: string | null
  authority: string | null
  owner_department: string | null
  file_hash: string | null
  upload_date: string | null
  chunks: number
}

// Logical document containing one or more versions
export interface DocumentInfo {
  document_id: string
  document_name: string
  store_type: 'shared' | 'private'
  is_shared: boolean
  versions: DocumentVersion[]
}

// Document list API response
export interface DocumentListResponse {
  status: string
  count: number
  documents: DocumentInfo[]
}

// ==================================================
// Helper Functions
// ==================================================

// Get JWT token from browser session
function getAuthHeaders(): HeadersInit {

  const token = sessionStorage.getItem('access_token')

  if (!token) {
    throw new Error('Authentication required. Please log in again.')
  }

  return {
    Authorization: `Bearer ${token}`,
  }
}


// ==================================================
// Login API
// ==================================================

export async function loginUser(
  credentials: LoginRequest
): Promise<LoginResponse> {

  const response = await fetch(`${API_BASE_URL}/login`, {
    method: 'POST',

    headers: {
      'Content-Type': 'application/json',
    },

    body: JSON.stringify(credentials),
  })

  if (!response.ok) {

    const errorData = await response.json().catch(() => null)

    throw new Error(
      errorData?.detail || 'Login failed. Please try again.'
    )
  }

  const data: LoginResponse = await response.json()

  return data
}


// ==================================================
// Get Chat Sessions
// ==================================================

export async function getSessions(): Promise<Record<string, SessionInfo>> {

  const response = await fetch(`${API_BASE_URL}/session/list`, {
    method: 'GET',

    headers: getAuthHeaders(),
  })

  if (!response.ok) {

    const errorData = await response.json().catch(() => null)

    throw new Error(
      errorData?.detail || 'Failed to retrieve chat sessions.'
    )
  }

  return await response.json()
}


// ==================================================
// Create New Chat Session
// ==================================================

export async function createSession(
  persona: string = 'default',
  title: string = 'New Chat'
): Promise<CreateSessionResponse> {

  const response = await fetch(`${API_BASE_URL}/session/create`, {
    method: 'POST',

    headers: {
      ...getAuthHeaders(),
      'Content-Type': 'application/json',
    },

    body: JSON.stringify({
      persona,
      title,
    }),
  })

  if (!response.ok) {

    const errorData = await response.json().catch(() => null)

    throw new Error(
      errorData?.detail || 'Failed to create a new chat.'
    )
  }

  return await response.json()
}


// ==================================================
// Get Messages for a Chat Session
// ==================================================

export async function getMessages(
  sessionId: string
): Promise<ChatMessage[]> {

  const response = await fetch(
    `${API_BASE_URL}/messages/${sessionId}`,
    {
      method: 'GET',

      headers: getAuthHeaders(),
    }
  )

  if (!response.ok) {

    const errorData = await response.json().catch(() => null)

    throw new Error(
      errorData?.detail || 'Failed to retrieve messages.'
    )
  }

  return await response.json()
}


// ==================================================
// Send Message to AI
// ==================================================

export async function sendMessage(
  sessionId: string,
  content: string
): Promise<SendMessageResponse> {

  const response = await fetch(`${API_BASE_URL}/message/send`, {
    method: 'POST',

    headers: {
      ...getAuthHeaders(),
      'Content-Type': 'application/json',
    },

    body: JSON.stringify({
      session_id: sessionId,
      content,
    }),
  })

  if (!response.ok) {

    const errorData = await response.json().catch(() => null)

    throw new Error(
      errorData?.detail || 'Failed to send message.'
    )
  }

  return await response.json()
}


// ==================================================
// Upload Document
// ==================================================

export async function uploadDocument(
  request: UploadDocumentRequest
): Promise<UploadDocumentResponse> {

  const formData = new FormData()

  // Required file
  formData.append('file', request.file)

  // Optional document metadata
  if (request.document_name) {
    formData.append('document_name', request.document_name)
  }

  if (request.document_version) {
    formData.append('document_version', request.document_version)
  }

  if (request.effective_date) {
    formData.append('effective_date', request.effective_date)
  }

  if (request.authority) {
    formData.append('authority', request.authority)
  }

  if (request.owner_department) {
    formData.append('owner_department', request.owner_department)
  }

  const response = await fetch(`${API_BASE_URL}/documents/upload`, {
    method: 'POST',

    headers: getAuthHeaders(),

    body: formData,
  })

  if (!response.ok) {

    const errorData = await response.json().catch(() => null)

    throw new Error(
      errorData?.detail || 'Failed to upload document.'
    )
  }

  return await response.json()
}


// ==================================================
// Get Uploaded Documents
// ==================================================

export async function getDocuments(): Promise<DocumentListResponse> {

  const response = await fetch(`${API_BASE_URL}/documents/list`, {
    method: 'GET',

    headers: getAuthHeaders(),
  })

  if (!response.ok) {

    const errorData = await response.json().catch(() => null)

    throw new Error(
      errorData?.detail || 'Failed to retrieve documents.'
    )
  }

  return await response.json()
}


// ==================================================
// Translate Text
// ==================================================

export async function translateText(
  text: string,
  targetLanguage: SupportedLanguage
): Promise<TranslateResponse> {

  // No API call is necessary when the target language is English
  if (targetLanguage === 'en') {
    return {
      original_text: text,
      translated_text: text,
      target_language: targetLanguage,
    }
  }

  const response = await fetch(`${API_BASE_URL}/translate`, {
    method: 'POST',

    headers: {
      ...getAuthHeaders(),
      'Content-Type': 'application/json',
    },

    body: JSON.stringify({
      text,
      target_language: targetLanguage,
    }),
  })

  if (!response.ok) {

    const errorData = await response.json().catch(() => null)

    throw new Error(
      errorData?.detail || 'Failed to translate text.'
    )
  }

  return await response.json()
}