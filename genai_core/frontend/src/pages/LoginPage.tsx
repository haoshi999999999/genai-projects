import { useState, useEffect, type FormEvent } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { loginUser } from '../services/api'
import './LoginPage.css'

type Language = 'en' | 'zh-CN' | 'zh-TW' | 'de' | 'fr' | 'ru'

const translations = {
  en: {
    title: 'GenAI Assistant',
    subtitle: 'Your intelligent assistant for work and learning.',
    email: 'Email',
    password: 'Password',
    emailPlaceholder: 'Enter your email',
    passwordPlaceholder: 'Enter your password',
    login: 'Login',
    register: 'Create an account',
    registrationSuccess: 'Registration successful. You can now log in.',
    darkMode: 'Dark Mode',
    lightMode: 'Light Mode',
    language: 'Language',
  },

  'zh-CN': {
    title: '生成式 AI 助手',
    subtitle: '您的工作与学习智能助手。',
    email: '邮箱',
    password: '密码',
    emailPlaceholder: '请输入邮箱',
    passwordPlaceholder: '请输入密码',
    login: '登录',
    register: '创建账户',
    registrationSuccess: '注册成功！现在可以登录了。',
    darkMode: '深色模式',
    lightMode: '浅色模式',
    language: '语言',
  },

  'zh-TW': {
    title: '生成式 AI 助理',
    subtitle: '您的工作與學習智慧助理。',
    email: '電子郵件',
    password: '密碼',
    emailPlaceholder: '請輸入電子郵件',
    passwordPlaceholder: '請輸入密碼',
    login: '登入',
    register: '建立帳戶',
    registrationSuccess: '註冊成功！現在可以登入了。',
    darkMode: '深色模式',
    lightMode: '淺色模式',
    language: '語言',
  },

  de: {
    title: 'GenAI-Assistent',
    subtitle: 'Ihr intelligenter Assistent für Arbeit und Lernen.',
    email: 'E-Mail',
    password: 'Passwort',
    emailPlaceholder: 'E-Mail eingeben',
    passwordPlaceholder: 'Passwort eingeben',
    login: 'Anmelden',
    register: 'Konto erstellen',
    registrationSuccess: 'Registrierung erfolgreich. Sie können sich jetzt anmelden.',
    darkMode: 'Dunkelmodus',
    lightMode: 'Hellmodus',
    language: 'Sprache',
  },

  fr: {
    title: 'Assistant GenAI',
    subtitle: 'Votre assistant intelligent pour le travail et l’apprentissage.',
    email: 'E-mail',
    password: 'Mot de passe',
    emailPlaceholder: 'Entrez votre e-mail',
    passwordPlaceholder: 'Entrez votre mot de passe',
    login: 'Connexion',
    register: 'Créer un compte',
    registrationSuccess: 'Inscription réussie. Vous pouvez maintenant vous connecter.',
    darkMode: 'Mode sombre',
    lightMode: 'Mode clair',
    language: 'Langue',
  },

  ru: {
    title: 'GenAI Ассистент',
    subtitle: 'Ваш интеллектуальный помощник для работы и обучения.',
    email: 'Электронная почта',
    password: 'Пароль',
    emailPlaceholder: 'Введите электронную почту',
    passwordPlaceholder: 'Введите пароль',
    login: 'Войти',
    register: 'Создать аккаунт',
    registrationSuccess: 'Регистрация прошла успешно. Теперь вы можете войти.',
    darkMode: 'Темный режим',
    lightMode: 'Светлый режим',
    language: 'Язык',
  },
}

const supportedLanguages: Language[] = [
  'en',
  'zh-CN',
  'zh-TW',
  'de',
  'fr',
  'ru',
]

function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')

  const navigate = useNavigate()
  const location = useLocation()

  // Read registration success message from RegisterPage
  const [registrationSuccess, setRegistrationSuccess] = useState('')

  // Load saved theme preference
  const [isDarkMode, setIsDarkMode] = useState(() => {
    return localStorage.getItem('theme') === 'dark'
  })

  // Load saved language preference
  const [language, setLanguage] = useState<Language>(() => {
    const savedLanguage = localStorage.getItem('language')

    if (
      savedLanguage &&
      supportedLanguages.includes(savedLanguage as Language)
    ) {
      return savedLanguage as Language
    }

    return 'en'
  })

  // Save theme preference and apply it application-wide
  useEffect(() => {
    const theme = isDarkMode ? 'dark' : 'light'

    localStorage.setItem('theme', theme)
    document.documentElement.dataset.theme = theme
  }, [isDarkMode])

  // Save language preference
  useEffect(() => {
    localStorage.setItem('language', language)
  }, [language])

  // Read and preserve registration success message
  useEffect(() => {
    const message = (
      location.state as { registrationSuccess?: string } | null
    )?.registrationSuccess

    if (message) {
      setRegistrationSuccess(message)

      navigate('/', {
        replace: true,
        state: null,
      })
    }
  }, [location.state, navigate])

  const t = translations[language]

  // Handle login form submission
  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    setErrorMessage('')
    setIsLoading(true)

    try {
      const result = await loginUser({
        email,
        password,
      })

      // Store JWT token for this browser session
      sessionStorage.setItem('access_token', result.access_token)

      console.log('Login successful!')

      // Redirect to the Chat page
      navigate('/chat')
    } catch (error) {
      // Remove any existing token if login fails
      sessionStorage.removeItem('access_token')

      if (error instanceof Error) {
        setErrorMessage(error.message)
      } else {
        setErrorMessage('An unexpected error occurred.')
      }
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className={`login-container ${isDarkMode ? 'dark' : ''}`}>
      {/* Left illustration panel */}

      {/* Right login panel */}
      <div className="login-box">

        {/* Top toolbar */}
        <div className="login-toolbar">

          {/* Language selector */}
          <div className="language-control">
            <label htmlFor="language-select">
              {t.language}
            </label>

            <select
              id="language-select"
              className="language-select"
              value={language}
              onChange={(event) =>
                setLanguage(event.target.value as Language)
              }
            >
              <option value="en">English</option>
              <option value="zh-CN">简体中文</option>
              <option value="zh-TW">繁體中文</option>
              <option value="de">Deutsch</option>
              <option value="fr">Français</option>
              <option value="ru">Русский</option>
            </select>
          </div>

          {/* Dark mode toggle */}
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setIsDarkMode((prev) => !prev)}
          >
            {isDarkMode ? t.lightMode : t.darkMode}
          </button>

        </div>

        {/* Page title */}
        <h1>{t.title}</h1>

        {/* Subtitle */}
        <p>{t.subtitle}</p>

        {/* Registration success message */}
        {registrationSuccess && (
          <div className="success-message" role="status">
            {registrationSuccess}
          </div>
        )}

        {/* Login form */}
        <form onSubmit={handleSubmit}>

          {/* Email */}
          <div className="form-group">
            <label htmlFor="email">
              {t.email}
            </label>

            <input
              id="email"
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder={t.emailPlaceholder}
              autoComplete="email"
              required
            />
          </div>

          {/* Password */}
          <div className="form-group">
            <label htmlFor="password">
              {t.password}
            </label>

            <input
              id="password"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder={t.passwordPlaceholder}
              autoComplete="current-password"
              required
            />
          </div>

          {/* Error message */}
          {errorMessage && (
            <div className="error-message" role="alert">
              {errorMessage}
            </div>
          )}

          {/* Login button */}
          <button
            type="submit"
            className="login-button"
            disabled={isLoading}
          >
            {isLoading ? 'Logging in...' : t.login}
          </button>

        </form>

        {/* Registration link */}
        <button
          type="button"
          className="register-link"
          onClick={() => navigate('/register')}
        >
          {t.register}
        </button>

      </div>
    </div>
  )
}

export default LoginPage