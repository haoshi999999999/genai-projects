import { useEffect, useState } from 'react'
import PageNavigation from '../components/PageNavigation'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import {
  getSessions,
  createSession,
  getMessages,
  sendMessage,
  translateText,
  type SessionInfo,
  type ChatMessage,
} from '../services/api'

import './ChatPage.css'

// ==========================================
// Language Configuration
// ==========================================

type Language = 'en' | 'zh-CN' | 'zh-TW' | 'de' | 'fr' | 'ru'

const supportedLanguages: Language[] = [
  'en',
  'zh-CN',
  'zh-TW',
  'de',
  'fr',
  'ru',
]

const translations = {
  en: {
    title: 'GenAI Assistant',
    newChat: '+ New Chat',
    creating: 'Creating...',
    recentConversations: 'Recent Conversations',
    noConversations: 'No conversations yet.',
    loadingConversations: 'Loading conversations...',
    logout: 'Logout',
    welcome: 'Welcome to GenAI Assistant',
    welcomeDescription:
      'Select a conversation or create a new chat to get started.',
    newConversation: 'New Conversation',
    newConversationDescription:
      'Start asking questions to your GenAI Assistant.',
    askAnything: 'Ask me anything...',
    send: 'Send',
    sending: 'Sending...',
    loadingMessages: 'Loading messages...',
    you: 'You',
    assistant: 'AI Assistant',
    thinking: 'Thinking...',
    closeConversation: 'Close conversation',
    confirmClose:
      'Are you sure you want to close this conversation?\n\nIt will be removed from your chat history in this browser. The conversation data will remain stored on the server.',
    loadError: 'Failed to load conversations.',
    messageError: 'Failed to load conversation messages.',
    createError: 'Failed to create a new chat.',
    sendError: 'Failed to send message.',
    unexpectedError: 'An unexpected error occurred.',
  },

  'zh-CN': {
    title: '生成式 AI 助手',
    newChat: '+ 新建聊天',
    creating: '正在创建...',
    recentConversations: '最近对话',
    noConversations: '暂无对话。',
    loadingConversations: '正在加载对话...',
    logout: '退出登录',
    welcome: '欢迎使用生成式 AI 助手',
    welcomeDescription: '请选择一个对话或创建新聊天以开始使用。',
    newConversation: '新对话',
    newConversationDescription: '开始向您的 AI 助手提问。',
    askAnything: '请输入您的问题...',
    send: '发送',
    sending: '发送中...',
    loadingMessages: '正在加载消息...',
    you: '您',
    assistant: 'AI 助手',
    thinking: '正在思考...',
    closeConversation: '关闭对话',
    confirmClose:
      '确定要关闭此对话吗？\n\n此对话将从当前浏览器的聊天记录中隐藏，但对话数据仍保存在服务器上。',
    loadError: '加载对话失败。',
    messageError: '加载对话消息失败。',
    createError: '创建新聊天失败。',
    sendError: '发送消息失败。',
    unexpectedError: '发生意外错误。',
  },

  'zh-TW': {
    title: '生成式 AI 助理',
    newChat: '+ 新增聊天',
    creating: '正在建立...',
    recentConversations: '最近對話',
    noConversations: '目前沒有對話。',
    loadingConversations: '正在載入對話...',
    logout: '登出',
    welcome: '歡迎使用生成式 AI 助理',
    welcomeDescription: '請選擇一個對話或建立新聊天以開始使用。',
    newConversation: '新對話',
    newConversationDescription: '開始向您的 AI 助理提問。',
    askAnything: '請輸入您的問題...',
    send: '傳送',
    sending: '傳送中...',
    loadingMessages: '正在載入訊息...',
    you: '您',
    assistant: 'AI 助理',
    thinking: '正在思考...',
    closeConversation: '關閉對話',
    confirmClose:
      '確定要關閉此對話嗎？\n\n此對話將從目前瀏覽器的聊天記錄中隱藏，但對話資料仍保存在伺服器上。',
    loadError: '載入對話失敗。',
    messageError: '載入對話訊息失敗。',
    createError: '建立新聊天失敗。',
    sendError: '傳送訊息失敗。',
    unexpectedError: '發生意外錯誤。',
  },

  de: {
    title: 'GenAI-Assistent',
    newChat: '+ Neuer Chat',
    creating: 'Wird erstellt...',
    recentConversations: 'Letzte Unterhaltungen',
    noConversations: 'Keine Unterhaltungen vorhanden.',
    loadingConversations: 'Unterhaltungen werden geladen...',
    logout: 'Abmelden',
    welcome: 'Willkommen beim GenAI-Assistenten',
    welcomeDescription:
      'Wählen Sie eine Unterhaltung aus oder erstellen Sie einen neuen Chat.',
    newConversation: 'Neue Unterhaltung',
    newConversationDescription:
      'Stellen Sie Ihrem GenAI-Assistenten Ihre erste Frage.',
    askAnything: 'Stellen Sie mir eine Frage...',
    send: 'Senden',
    sending: 'Wird gesendet...',
    loadingMessages: 'Nachrichten werden geladen...',
    you: 'Sie',
    assistant: 'KI-Assistent',
    thinking: 'Denkt nach...',
    closeConversation: 'Unterhaltung schließen',
    confirmClose:
      'Möchten Sie diese Unterhaltung wirklich schließen?\n\nSie wird aus dem Chatverlauf dieses Browsers ausgeblendet. Die Daten bleiben auf dem Server gespeichert.',
    loadError: 'Unterhaltungen konnten nicht geladen werden.',
    messageError: 'Nachrichten konnten nicht geladen werden.',
    createError: 'Neuer Chat konnte nicht erstellt werden.',
    sendError: 'Nachricht konnte nicht gesendet werden.',
    unexpectedError: 'Ein unerwarteter Fehler ist aufgetreten.',
  },

  fr: {
    title: 'Assistant GenAI',
    newChat: '+ Nouvelle conversation',
    creating: 'Création en cours...',
    recentConversations: 'Conversations récentes',
    noConversations: 'Aucune conversation.',
    loadingConversations: 'Chargement des conversations...',
    logout: 'Déconnexion',
    welcome: 'Bienvenue dans Assistant GenAI',
    welcomeDescription:
      'Sélectionnez une conversation ou créez-en une nouvelle pour commencer.',
    newConversation: 'Nouvelle conversation',
    newConversationDescription:
      'Commencez à poser des questions à votre assistant GenAI.',
    askAnything: 'Posez-moi une question...',
    send: 'Envoyer',
    sending: 'Envoi en cours...',
    loadingMessages: 'Chargement des messages...',
    you: 'Vous',
    assistant: 'Assistant IA',
    thinking: 'Réflexion en cours...',
    closeConversation: 'Fermer la conversation',
    confirmClose:
      'Voulez-vous vraiment fermer cette conversation ?\n\nElle sera masquée de votre historique dans ce navigateur. Les données resteront enregistrées sur le serveur.',
    loadError: 'Impossible de charger les conversations.',
    messageError: 'Impossible de charger les messages.',
    createError: 'Impossible de créer une nouvelle conversation.',
    sendError: 'Impossible d’envoyer le message.',
    unexpectedError: 'Une erreur inattendue s’est produite.',
  },

  ru: {
    title: 'GenAI Ассистент',
    newChat: '+ Новый чат',
    creating: 'Создание...',
    recentConversations: 'Недавние беседы',
    noConversations: 'Бесед пока нет.',
    loadingConversations: 'Загрузка бесед...',
    logout: 'Выйти',
    welcome: 'Добро пожаловать в GenAI Ассистент',
    welcomeDescription:
      'Выберите беседу или создайте новый чат, чтобы начать.',
    newConversation: 'Новая беседа',
    newConversationDescription:
      'Начните задавать вопросы своему AI-ассистенту.',
    askAnything: 'Задайте мне вопрос...',
    send: 'Отправить',
    sending: 'Отправка...',
    loadingMessages: 'Загрузка сообщений...',
    you: 'Вы',
    assistant: 'AI-ассистент',
    thinking: 'Думаю...',
    closeConversation: 'Закрыть беседу',
    confirmClose:
      'Вы действительно хотите закрыть эту беседу?\n\nОна будет скрыта из истории чатов в этом браузере. Данные беседы останутся на сервере.',
    loadError: 'Не удалось загрузить беседы.',
    messageError: 'Не удалось загрузить сообщения.',
    createError: 'Не удалось создать новый чат.',
    sendError: 'Не удалось отправить сообщение.',
    unexpectedError: 'Произошла непредвиденная ошибка.',
  },
}


// ==========================================
// Browser Storage Configuration
// ==========================================

const HIDDEN_SESSIONS_KEY = 'genai_hidden_sessions'

const TRANSLATION_CACHE_KEY = 'genai_translation_cache_v1'

// Prevent duplicate translation requests for the same title
const translationRequests = new Map<string, Promise<string>>()


function getSavedLanguage(): Language {
  const savedLanguage = localStorage.getItem('language')

  if (
    savedLanguage &&
    supportedLanguages.includes(savedLanguage as Language)
  ) {
    return savedLanguage as Language
  }

  return 'en'
}


// Load cached translations from localStorage
function getTranslationCache(): Record<string, string> {
  try {
    const savedCache = localStorage.getItem(TRANSLATION_CACHE_KEY)

    if (!savedCache) {
      return {}
    }

    const parsedCache: unknown = JSON.parse(savedCache)

    if (
      parsedCache !== null &&
      typeof parsedCache === 'object' &&
      !Array.isArray(parsedCache)
    ) {
      return parsedCache as Record<string, string>
    }

    return {}

  } catch {
    return {}
  }
}


// Save one translated title to localStorage
function saveTranslationToCache(
  cacheKey: string,
  translatedText: string
) {
  try {
    const currentCache = getTranslationCache()

    currentCache[cacheKey] = translatedText

    localStorage.setItem(
      TRANSLATION_CACHE_KEY,
      JSON.stringify(currentCache)
    )

  } catch {
    console.error('Unable to save translation cache.')
  }
}


// Generate a unique cache key for each session, language, and original title
function getTitleCacheKey(
  sessionId: string,
  language: Language,
  originalTitle: string
): string {
  return JSON.stringify([
    sessionId,
    language,
    originalTitle,
  ])
}


// ==========================================
// Main Chat Component
// ==========================================

function ChatPage() {

  // ==========================================
  // Language
  // ==========================================

  const [language] = useState<Language>(getSavedLanguage)

  const t = translations[language]


  // ==========================================
  // State Management
  // ==========================================

  const [sessions, setSessions] = useState<Record<string, SessionInfo>>({})

  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)

  const [messages, setMessages] = useState<ChatMessage[]>([])

  const [inputMessage, setInputMessage] = useState('')

  const [hiddenSessionIds, setHiddenSessionIds] = useState<string[]>(() => {
    try {
      const savedIds = localStorage.getItem(HIDDEN_SESSIONS_KEY)

      if (!savedIds) {
        return []
      }

      const parsedIds: unknown = JSON.parse(savedIds)

      if (Array.isArray(parsedIds)) {
        return parsedIds.filter(
          (id): id is string => typeof id === 'string'
        )
      }

      return []
    } catch {
      return []
    }
  })

  const [translatedTitles, setTranslatedTitles] = useState<
    Record<string, string>
  >(getTranslationCache)

  const [isLoading, setIsLoading] = useState(true)
  const [isLoadingMessages, setIsLoadingMessages] = useState(false)
  const [isCreatingChat, setIsCreatingChat] = useState(false)
  const [isSending, setIsSending] = useState(false)

  const [errorMessage, setErrorMessage] = useState('')


  // ==========================================
  // Persist Hidden Conversations
  // ==========================================

  useEffect(() => {
    try {
      localStorage.setItem(
        HIDDEN_SESSIONS_KEY,
        JSON.stringify(hiddenSessionIds)
      )
    } catch {
      console.error('Unable to save hidden conversations.')
    }
  }, [hiddenSessionIds])


  // ==========================================
  // Translate Conversation Titles
  // ==========================================

  useEffect(() => {

    // English titles do not need translation
    if (language === 'en') {
      return
    }

    let isActive = true

    const translateSessionTitles = async () => {

      const translationTasks = Object.entries(sessions).map(
        async ([sessionId, session]) => {

          const originalTitle = session.title?.trim()

          if (!originalTitle) {
            return
          }

          const cacheKey = getTitleCacheKey(
            sessionId,
            language,
            originalTitle
          )

          // Already translated and cached
          if (translatedTitles[cacheKey]) {
            return
          }

          try {

            let translationPromise = translationRequests.get(cacheKey)

            // Reuse an existing request if one is already running
            if (!translationPromise) {

              translationPromise = translateText(
                originalTitle,
                language
              ).then((result) => result.translated_text)

              translationRequests.set(
                cacheKey,
                translationPromise
              )
            }

            const translatedText = await translationPromise

            if (translatedText) {

              saveTranslationToCache(
                cacheKey,
                translatedText
              )

              if (isActive) {
                setTranslatedTitles((previousTitles) => ({
                  ...previousTitles,
                  [cacheKey]: translatedText,
                }))
              }
            }

          } catch (error) {

            // Keep the original title if translation fails
            console.error(
              `Unable to translate conversation ${sessionId}:`,
              error
            )

          } finally {

            // Remove completed request from the in-flight request map
            translationRequests.delete(cacheKey)
          }
        }
      )

      await Promise.all(translationTasks)
    }

    translateSessionTitles()

    return () => {
      isActive = false
    }

  }, [sessions, language, translatedTitles])


  // ==========================================
  // Load Chat Sessions
  // ==========================================

  useEffect(() => {
    const loadSessions = async () => {
      try {
        setIsLoading(true)
        setErrorMessage('')

        const data = await getSessions()

        setSessions(data)

      } catch (error) {
        if (error instanceof Error) {
          setErrorMessage(error.message)
        } else {
          setErrorMessage(t.loadError)
        }

      } finally {
        setIsLoading(false)
      }
    }

    loadSessions()
  }, [])


  // ==========================================
  // Select Existing Conversation
  // ==========================================

  const handleSelectSession = async (sessionId: string) => {
    try {
      setActiveSessionId(sessionId)

      setMessages([])

      setInputMessage('')

      setIsLoadingMessages(true)

      setErrorMessage('')

      const data = await getMessages(sessionId)

      setMessages(data)

    } catch (error) {
      if (error instanceof Error) {
        setErrorMessage(error.message)
      } else {
        setErrorMessage(t.messageError)
      }

    } finally {
      setIsLoadingMessages(false)
    }
  }


  // ==========================================
  // Close Conversation
  // ==========================================

  const handleCloseSession = (sessionId: string) => {

    const confirmed = window.confirm(t.confirmClose)

    if (!confirmed) {
      return
    }

    setHiddenSessionIds((previousIds) => {
      if (previousIds.includes(sessionId)) {
        return previousIds
      }

      return [...previousIds, sessionId]
    })

    if (activeSessionId === sessionId) {
      setActiveSessionId(null)
      setMessages([])
      setInputMessage('')
      setErrorMessage('')
      setIsLoadingMessages(false)
    }
  }


  // ==========================================
  // Create New Chat
  // ==========================================

  const handleNewChat = async () => {
    try {
      setIsCreatingChat(true)

      setErrorMessage('')

      const result = await createSession()

      const updatedSessions = await getSessions()

      setSessions(updatedSessions)

      setActiveSessionId(result.session_id)

      setMessages([])

      setInputMessage('')

    } catch (error) {
      if (error instanceof Error) {
        setErrorMessage(error.message)
      } else {
        setErrorMessage(t.createError)
      }

    } finally {
      setIsCreatingChat(false)
    }
  }


  // ==========================================
  // Send Message to AI
  // ==========================================

  const handleSendMessage = async () => {

    if (
      !inputMessage.trim() ||
      !activeSessionId ||
      isSending ||
      isLoadingMessages
    ) {
      return
    }

    const userMessage = inputMessage.trim()

    const currentSessionId = activeSessionId

    try {
      setIsSending(true)

      setErrorMessage('')

      setInputMessage('')

      setMessages((previousMessages) => [
        ...previousMessages,
        {
          role: 'user',
          content: userMessage,
        },
      ])

      const result = await sendMessage(
        currentSessionId,
        userMessage
      )

      if (currentSessionId === activeSessionId) {
        setMessages((previousMessages) => [
          ...previousMessages,
          {
            role: 'assistant',
            content: result.reply,
          },
        ])
      }

      const updatedSessions = await getSessions()

      setSessions(updatedSessions)

    } catch (error) {

      if (error instanceof Error) {
        setErrorMessage(error.message)
      } else {
        setErrorMessage(t.sendError)
      }

      try {
        const updatedMessages = await getMessages(currentSessionId)

        if (currentSessionId === activeSessionId) {
          setMessages(updatedMessages)
        }
      } catch {
        // Keep the original error message.
      }

    } finally {
      setIsSending(false)
    }
  }


  // ==========================================
  // Visible Conversation List
  // ==========================================

  const visibleSessions = Object.entries(sessions).filter(
    ([sessionId]) => !hiddenSessionIds.includes(sessionId)
  )


  // ==========================================
  // Get Display Title
  // ==========================================

  const getDisplayTitle = (
    sessionId: string,
    originalTitle: string
  ): string => {

    if (language === 'en') {
      return originalTitle
    }

    const cacheKey = getTitleCacheKey(
      sessionId,
      language,
      originalTitle
    )

    return translatedTitles[cacheKey] || originalTitle
  }


  // ==========================================
  // Render UI
  // ==========================================

  return (
    <>
      {/* Shared navigation */}
      <PageNavigation logoutLabel={t.logout} />

      {/* Chat interface */}
      <div className="chat-container">

        {/* Sidebar */}
        <aside className="chat-sidebar">

          <div className="sidebar-header">
            <h2>{t.title}</h2>
          </div>


          {/* New Chat */}
          <button
            className="new-chat-button"
            onClick={handleNewChat}
            disabled={isCreatingChat}
          >
            {isCreatingChat ? t.creating : t.newChat}
          </button>


          {/* Conversation History */}
          <div className="chat-history">

            <h3>{t.recentConversations}</h3>

            {isLoading && (
              <p>{t.loadingConversations}</p>
            )}

            {errorMessage && (
              <p className="error-message">
                {errorMessage}
              </p>
            )}

            {!isLoading &&
              visibleSessions.map(([sessionId, session]) => {

                const displayTitle = getDisplayTitle(
                  sessionId,
                  session.title
                )

                return (
                  <div
                    key={sessionId}
                    className={`history-item-container ${
                      activeSessionId === sessionId ? 'active' : ''
                    }`}
                  >

                    <button
                      type="button"
                      className="history-item"
                      onClick={() => handleSelectSession(sessionId)}
                      disabled={isSending}
                      title={displayTitle}
                    >
                      {displayTitle}
                    </button>

                    <button
                      type="button"
                      className="close-session-button"
                      onClick={() => handleCloseSession(sessionId)}
                      disabled={isSending}
                      aria-label={`${t.closeConversation}: ${displayTitle}`}
                      title={t.closeConversation}
                    >
                      ×
                    </button>

                  </div>
                )
              })
            }

            {!isLoading &&
              visibleSessions.length === 0 && (
                <p className="empty-history">
                  {t.noConversations}
                </p>
              )}

          </div>

        </aside>


        {/* Main Chat Area */}
        <main className="chat-main">

          <header className="chat-header">
            <h2>{t.title}</h2>
          </header>


          {/* Welcome Screen */}
          {!activeSessionId && (

            <section className="chat-welcome">

              <h1>{t.welcome}</h1>

              <p>{t.welcomeDescription}</p>

            </section>

          )}


          {/* Conversation Messages */}
          {activeSessionId && (

            <section className="message-list">

              {isLoadingMessages && (
                <p>{t.loadingMessages}</p>
              )}


              {!isLoadingMessages && messages.length === 0 && (

                <div className="chat-welcome">

                  <h1>{t.newConversation}</h1>

                  <p>{t.newConversationDescription}</p>

                </div>

              )}


              {!isLoadingMessages && messages.map((message, index) => (

                <div
                  key={index}
                  className={`message-item ${message.role}`}
                >

                  <div className="message-role">
                    {message.role === 'user' ? t.you : t.assistant}
                  </div>

                  <div className="message-content">
                  {message.role === 'assistant' ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {message.content}
                    </ReactMarkdown>
                  ) : (
                    message.content
                  )}
                </div>

                </div>

              ))}


              {/* AI Thinking Status */}
              {isSending && (

                <div className="message-item assistant">

                  <div className="message-role">
                    {t.assistant}
                  </div>

                  <div className="message-content">
                    {t.thinking}
                  </div>

                </div>

              )}

            </section>

          )}


          {/* Error Message */}
          {errorMessage && activeSessionId && (

            <div className="chat-error">
              {errorMessage}
            </div>

          )}


          {/* Message Input */}
          <div className="chat-input-container">

            <input
              type="text"
              placeholder={t.askAnything}
              value={inputMessage}
              onChange={(event) => setInputMessage(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  handleSendMessage()
                }
              }}
              disabled={
                !activeSessionId ||
                isSending ||
                isLoadingMessages
              }
            />

            <button
              onClick={handleSendMessage}
              disabled={
                !activeSessionId ||
                !inputMessage.trim() ||
                isSending ||
                isLoadingMessages
              }
            >
              {isSending ? t.sending : t.send}
            </button>

          </div>

        </main>

      </div>
    </>
  )
}

export default ChatPage
