
const API_BASE_URL = ''

export interface SourceReference {
  document_id: string
  filename: string | null
  chunk_ids: string[]
}

export interface QuestionResponse {
  question: string
  answer: string
  sources: SourceReference[]
}

export interface Document {
  document_id: string
  filename: string
  file_size: number
  status: string
  total_chunks: number
  created_at: string
}

interface DocumentListResponse {
  documents: Document[]
  limit: number
  offset: number
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => null)

    const message =
      errorData?.detail
        ? typeof errorData.detail === 'string'
          ? errorData.detail
          : JSON.stringify(errorData.detail)
        : `Request failed: ${response.status}`

    throw new Error(message)
  }

  return response.json()
}

export async function uploadDocument(file: File) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(
    `${API_BASE_URL}/api/v1/documents/upload`,
    {
      method: 'POST',
      body: formData,
    }
  )

  return handleResponse<unknown>(response)
}

export async function fetchDocuments(): Promise<DocumentListResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/documents`
  )

  return handleResponse<DocumentListResponse>(response)
}

export async function askQuestion(
  question: string,
  documentId?: string
): Promise<QuestionResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/questions`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        ...(documentId ? { document_id: documentId } : {}),
      }),
    }
  )

  return handleResponse<QuestionResponse>(response)
}


export async function deleteDocument(documentId: string) {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/documents/${documentId}`,
    {
      method: 'DELETE',
    }
  )

  return handleResponse<{
    document_id: string
    deleted_chunks: number
    message: string
  }>(response)
}
