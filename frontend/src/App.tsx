
import { useEffect, useRef, useState } from 'react'
import {
  uploadDocument,
  fetchDocuments,
  deleteDocument,
  askQuestion,
  type Document,
  type QuestionResponse,
} from './services/api'
import './App.css'

type Theme = 'light' | 'dark'

interface ChatMessage {
  id: string
  question: string
  answer: string
  sources: QuestionResponse['sources']
  status: 'loading' | 'success' | 'error'
  error?: string
  documentId?: string
  documentName?: string
}

const suggestions = [
  'Summarize my document',
  'Explain the key concepts',
  'What are the main findings?',
]

function App() {
  const [question, setQuestion] = useState('')

  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return localStorage.getItem('docassistant-theme') === 'dark'
        ? 'dark'
        : 'light'
    } catch {
      return 'light'
    }
  })

  const [uploading, setUploading] = useState(false)
  const [uploadMessage, setUploadMessage] = useState('')
  const [uploadError, setUploadError] = useState('')

  const [documents, setDocuments] = useState<Document[]>([])
  const [loadingDocuments, setLoadingDocuments] = useState(true)
  const [documentError, setDocumentError] = useState('')

  const [selectedDocumentId, setSelectedDocumentId] =
    useState<string | null>(null)

  const [deletingDocumentId, setDeletingDocumentId] =
    useState<string | null>(null)

  const [confirmDeleteDocumentId, setConfirmDeleteDocumentId] =
    useState<string | null>(null)

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [asking, setAsking] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const chatEndRef = useRef<HTMLDivElement>(null)
  const composerRef = useRef<HTMLTextAreaElement>(null)

  const selectedDocument = documents.find(
    (document) => document.document_id === selectedDocumentId
  )

  const documentToDelete = documents.find(
    (document) =>
      document.document_id === confirmDeleteDocumentId
  )

  const totalChunks = documents.reduce(
    (total, document) => total + document.total_chunks,
    0
  )

  // Persist the selected theme.
  useEffect(() => {
    try {
      localStorage.setItem('docassistant-theme', theme)
    } catch {
      // The theme still works for the current session.
    }
  }, [theme])

  // Scroll to the latest conversation message.
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'end',
    })
  }, [messages])

  // Load documents.
  async function loadDocuments() {
    try {
      setDocumentError('')

      const result = await fetchDocuments()
      setDocuments(result.documents)

      setSelectedDocumentId((currentId) => {
        if (
          currentId &&
          !result.documents.some(
            (document) => document.document_id === currentId
          )
        ) {
          return null
        }

        return currentId
      })
    } catch (error) {
      setDocumentError(
        error instanceof Error
          ? error.message
          : 'Failed to load documents.'
      )
    } finally {
      setLoadingDocuments(false)
    }
  }

  useEffect(() => {
    loadDocuments()
  }, [])

  // Upload PDF.
  async function handleFileUpload(
    event: React.ChangeEvent<HTMLInputElement>
  ) {
    const file = event.target.files?.[0]

    if (!file) return

    setUploadMessage('')
    setUploadError('')

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadError('Please select a PDF file.')
      event.target.value = ''
      return
    }

    try {
      setUploading(true)

      await uploadDocument(file)

      setUploadMessage(`${file.name} uploaded successfully.`)

      await loadDocuments()
    } catch (error) {
      setUploadError(
        error instanceof Error
          ? error.message
          : 'Upload failed. Please try again.'
      )
    } finally {
      setUploading(false)
      event.target.value = ''
    }
  }

  // Select or deselect a document.
  function handleDocumentSelect(documentId: string) {
    setSelectedDocumentId((currentId) =>
      currentId === documentId ? null : documentId
    )
  }

  // Send a question to the RAG API.
  async function sendQuestion(
    messageId: string,
    messageQuestion: string,
    documentId?: string
  ) {
    setAsking(true)

    try {
      const result = await askQuestion(
        messageQuestion,
        documentId
      )

      setMessages((currentMessages) =>
        currentMessages.map((message) =>
          message.id === messageId
            ? {
                ...message,
                answer: result.answer,
                sources: result.sources,
                status: 'success',
                error: undefined,
              }
            : message
        )
      )
    } catch (error) {
      setMessages((currentMessages) =>
        currentMessages.map((message) =>
          message.id === messageId
            ? {
                ...message,
                status: 'error',
                error:
                  error instanceof Error
                    ? error.message
                    : 'Failed to get an answer. Please try again.',
              }
            : message
        )
      )
    } finally {
      setAsking(false)
    }
  }

  // Submit a question.
  async function handleQuestionSubmit(
    event: React.FormEvent<HTMLFormElement>
  ) {
    event.preventDefault()

    const trimmedQuestion = question.trim()

    if (!trimmedQuestion || asking) return

    const messageId =
      typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random()}`

    const newMessage: ChatMessage = {
      id: messageId,
      question: trimmedQuestion,
      answer: '',
      sources: [],
      status: 'loading',
      documentId: selectedDocumentId ?? undefined,
      documentName: selectedDocument?.filename,
    }

    setMessages((currentMessages) => [
      ...currentMessages,
      newMessage,
    ])

    setQuestion('')

    await sendQuestion(
      messageId,
      trimmedQuestion,
      selectedDocumentId ?? undefined
    )
  }

  // Retry a failed question.
  async function retryQuestion(message: ChatMessage) {
    if (asking) return

    setMessages((currentMessages) =>
      currentMessages.map((item) =>
        item.id === message.id
          ? {
              ...item,
              status: 'loading',
              error: undefined,
            }
          : item
      )
    )

    await sendQuestion(
      message.id,
      message.question,
      message.documentId
    )
  }

  // Start a fresh conversation.
  function handleNewChat() {
    if (asking) return

    setMessages([])
    setQuestion('')
    composerRef.current?.focus()
  }

  // Select a suggested question.
  function handleSuggestionClick(value: string) {
    setQuestion(value)
    composerRef.current?.focus()
  }

  // Delete confirmation.
  function openDeleteConfirmation(documentId: string) {
    setConfirmDeleteDocumentId(documentId)
  }

  function cancelDocumentDelete() {
    if (deletingDocumentId) return

    setConfirmDeleteDocumentId(null)
  }

  // Delete document.
  async function handleDocumentDelete() {
    if (!confirmDeleteDocumentId) return

    const documentId = confirmDeleteDocumentId

    try {
      setDeletingDocumentId(documentId)
      setDocumentError('')
      setUploadMessage('')
      setUploadError('')

      await deleteDocument(documentId)

      setDocuments((currentDocuments) =>
        currentDocuments.filter(
          (document) => document.document_id !== documentId
        )
      )

      if (selectedDocumentId === documentId) {
        setSelectedDocumentId(null)
      }

      setUploadMessage('Document deleted successfully.')
    } catch (error) {
      setDocumentError(
        error instanceof Error
          ? error.message
          : 'Failed to delete document.'
      )
    } finally {
      setDeletingDocumentId(null)
      setConfirmDeleteDocumentId(null)
    }
  }

  return (
    <div className="app" data-theme={theme}>

      {/* Sidebar */}
      <aside className="sidebar">

        <div className="brand">
          <div className="brand-icon">✦</div>

          <div className="brand-text">
            <strong>DocAssistant</strong>
            <span>INTELLIGENT WORKSPACE</span>
          </div>
        </div>

        <button
          className="new-chat-button"
          onClick={handleNewChat}
          disabled={asking}
        >
          <span>＋</span>
          New conversation
        </button>

        <button
          className="upload-button"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          <span className="upload-icon">↑</span>
          <span>
            {uploading ? 'Uploading...' : 'Upload PDF'}
          </span>
          <span className="upload-format">PDF</span>
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          hidden
          onChange={handleFileUpload}
        />

        <div className="library-section">

          <div className="library-heading">
            <span>YOUR LIBRARY</span>
            <span className="document-count">
              {documents.length}
            </span>
          </div>

          {loadingDocuments && (
            <p className="sidebar-message">
              Loading your library...
            </p>
          )}

          {documentError && (
            <p className="error-message">
              {documentError}
            </p>
          )}

          {!loadingDocuments &&
            !documentError &&
            documents.length === 0 && (
              <div className="empty-library">
                <span>▤</span>
                <p>No documents yet</p>
                <small>Upload a PDF to get started.</small>
              </div>
            )}

          <div className="document-list">
            {documents.map((document) => (
              <div
                key={document.document_id}
                className={`document-card ${
                  selectedDocumentId === document.document_id
                    ? 'selected'
                    : ''
                }`}
              >
                <button
                  className="document-select"
                  onClick={() =>
                    handleDocumentSelect(document.document_id)
                  }
                  aria-pressed={
                    selectedDocumentId === document.document_id
                  }
                  title={document.filename}
                >
                  <span className="document-icon">▤</span>

                  <span className="document-content">
                    <span className="document-name">
                      {document.filename}
                    </span>

                    <span className="document-meta">
                      {document.total_chunks} chunks
                      <span className="meta-separator">·</span>
                      <span className="status-dot" />
                      {document.status}
                    </span>
                  </span>

                  {selectedDocumentId === document.document_id && (
                    <span className="selected-indicator">
                      ✓
                    </span>
                  )}
                </button>

                <div className="document-actions">
                  <span>
                    {selectedDocumentId === document.document_id
                      ? 'Selected for chat'
                      : 'Click to select'}
                  </span>

                  <button
                    className="delete-button"
                    onClick={() =>
                      openDeleteConfirmation(document.document_id)
                    }
                    disabled={
                      deletingDocumentId === document.document_id
                    }
                    title="Delete document"
                    aria-label={`Delete ${document.filename}`}
                  >
                    {deletingDocumentId === document.document_id
                      ? '...'
                      : 'Delete'}
                  </button>
                </div>
              </div>
            ))}
          </div>

        </div>

        {uploadMessage && (
          <p className="success-message">{uploadMessage}</p>
        )}

        {uploadError && (
          <p className="error-message">{uploadError}</p>
        )}

        <div className="sidebar-bottom">
          <div className="library-stats">
            <div className="stat-icon">▤</div>
            <div>
              <strong>{documents.length} documents</strong>
              <span>{totalChunks} indexed chunks</span>
            </div>
          </div>

          <div className="sidebar-footer">
            <span className="footer-status-dot" />
            RAG workspace active
          </div>
        </div>

      </aside>

      {/* Main workspace */}
      <main className="main-content">

        <header className="topbar">

          <div className="topbar-title">
            <span className="topbar-eyebrow">
              YOUR WORKSPACE
            </span>

            <h2>Document Intelligence</h2>
          </div>

          <div className="topbar-actions">

            {selectedDocument && (
              <div className="active-document">
                <span>▤</span>
                <span>{selectedDocument.filename}</span>
              </div>
            )}

            <button
              className="theme-toggle"
              onClick={() =>
                setTheme((current) =>
                  current === 'light' ? 'dark' : 'light'
                )
              }
              aria-label={`Switch to ${
                theme === 'light' ? 'dark' : 'light'
              } theme`}
              title="Toggle theme"
            >
              {theme === 'light' ? '☾' : '☀'}
            </button>

            <div className="assistant-status">
              <span />
              AI READY
            </div>

          </div>

        </header>

        <section className="chat-area">

          {/* Welcome screen */}
          {messages.length === 0 && (
            <div className="welcome-screen">

              <div className="welcome-badge">
                <span>✦</span>
                YOUR DOCUMENTS, UNDERSTOOD
              </div>

              <h1>
                Hey Dharan<span>.</span>
                <br />
                What are we exploring today?
              </h1>

              <p className="welcome-description">
                Your intelligent document workspace is ready.
                Ask questions, discover insights, and explore
                the knowledge inside your PDFs.
              </p>

              {selectedDocument && (
                <div className="welcome-document">
                  <span>▤</span>
                  Searching in <strong>{selectedDocument.filename}</strong>
                </div>
              )}

              <div className="suggestion-section">
                <span className="suggestion-label">
                  GET STARTED WITH
                </span>

                <div className="suggestion-list">
                  {suggestions.map((suggestion) => (
                    <button
                      key={suggestion}
                      className="suggestion-card"
                      onClick={() =>
                        handleSuggestionClick(suggestion)
                      }
                      disabled={documents.length === 0}
                    >
                      <span>✧</span>
                      {suggestion}
                      <span className="suggestion-arrow">↗</span>
                    </button>
                  ))}
                </div>
              </div>

              {documents.length === 0 && (
                <p className="welcome-hint">
                  Upload a PDF from the sidebar to begin.
                </p>
              )}

            </div>
          )}

          {/* Conversation */}
          {messages.length > 0 && (
            <div className="conversation">

              {messages.map((message) => (
                <div
                  className="message-group"
                  key={message.id}
                >

                  <div className="user-message">
                    <div className="user-avatar">D</div>

                    <div className="user-message-content">
                      <span className="message-label">YOU</span>
                      <p>{message.question}</p>
                    </div>
                  </div>

                  <div className="assistant-message">

                    <div className="assistant-avatar">
                      ✦
                    </div>

                    <div className="assistant-message-content">

                      <div className="assistant-message-heading">
                        <strong>DocAssistant</strong>
                        <span>AI RESPONSE</span>
                      </div>

                      {message.documentName && (
                        <div className="response-context">
                          <span>▤</span>
                          {message.documentName}
                        </div>
                      )}

                      {message.status === 'loading' && (
                        <div className="loading-indicator">
                          <span className="loading-dot" />
                          <span className="loading-dot" />
                          <span className="loading-dot" />
                          <span>Analyzing your documents...</span>
                        </div>
                      )}

                      {message.status === 'error' && (
                        <div className="chat-error">
                          <p>{message.error}</p>

                          <button
                            className="retry-button"
                            onClick={() => retryQuestion(message)}
                            disabled={asking}
                          >
                            ↻ Try again
                          </button>
                        </div>
                      )}

                      {message.status === 'success' && (
                        <>
                          <div className="answer-text">
                            {message.answer}
                          </div>

                          <div className="sources-section">
                            <div className="sources-heading">
                              <span>◈</span>
                              <strong>Sources</strong>
                              <span className="sources-count">
                                {message.sources.length}
                              </span>
                            </div>

                            {message.sources.length > 0 ? (
                              <div className="source-list">
                                {message.sources.map((source, index) => (
                                  <div
                                    className="source-item"
                                    key={`${message.id}-${source.document_id}-${index}`}
                                  >
                                    <div className="source-item-top">
                                      <span className="source-file-icon">
                                        ▤
                                      </span>

                                      <strong>
                                        {source.filename ||
                                          'Unknown document'}
                                      </strong>
                                    </div>

                                    <div className="source-details">
                                      {source.chunk_ids.length} retrieved chunks
                                    </div>

                                    {source.chunk_ids.length > 0 && (
                                      <details className="source-chunks">
                                        <summary>
                                          View chunk references
                                        </summary>

                                        <ul>
                                          {source.chunk_ids.map((chunkId) => (
                                            <li key={chunkId}>
                                              {chunkId}
                                            </li>
                                          ))}
                                        </ul>
                                      </details>
                                    )}
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <p className="empty-text">
                                No source references returned.
                              </p>
                            )}
                          </div>
                        </>
                      )}

                    </div>

                  </div>

                </div>
              ))}

              <div ref={chatEndRef} />

            </div>
          )}

        </section>

        {/* Composer */}
        <div className="composer-area">

          {selectedDocument && messages.length > 0 && (
            <div className="composer-context">
              <span>▤</span>
              <span>{selectedDocument.filename}</span>
              <span>·</span>
              <span>Document selected</span>
            </div>
          )}

          <form
            className="question-form"
            onSubmit={handleQuestionSubmit}
          >
            <textarea
              ref={composerRef}
              placeholder="Ask DocAssistant anything about your documents..."
              value={question}
              rows={1}
              disabled={asking}
              onChange={(event) =>
                setQuestion(event.target.value)
              }
              onKeyDown={(event) => {
                if (
                  event.key === 'Enter' &&
                  !event.shiftKey &&
                  !event.nativeEvent.isComposing
                ) {
                  event.preventDefault()
                  event.currentTarget.form?.requestSubmit()
                }
              }}
            />

            <button
              type="submit"
              disabled={asking || !question.trim()}
              aria-label="Send message"
              title="Send message"
            >
              {asking ? '…' : '↑'}
            </button>
          </form>

          <p className="composer-hint">
            Enter to send <span>·</span> Shift + Enter for a new line
          </p>

          <p className="disclaimer">
            DocAssistant can make mistakes. Verify important information
            against your original documents.
          </p>

        </div>

      </main>

      {/* Delete confirmation */}
      {confirmDeleteDocumentId && documentToDelete && (
        <div className="modal-overlay">
          <div
            className="delete-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-modal-title"
          >
            <div className="modal-icon">!</div>

            <h3 id="delete-modal-title">
              Delete document?
            </h3>

            <p>
              Are you sure you want to delete{' '}
              <strong>{documentToDelete.filename}</strong>?
            </p>

            <p className="modal-warning">
              This action cannot be undone.
            </p>

            <div className="modal-actions">
              <button
                className="cancel-button"
                onClick={cancelDocumentDelete}
                disabled={deletingDocumentId !== null}
              >
                Cancel
              </button>

              <button
                className="confirm-delete-button"
                onClick={handleDocumentDelete}
                disabled={deletingDocumentId !== null}
              >
                {deletingDocumentId
                  ? 'Deleting...'
                  : 'Delete document'}
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}

export default App
