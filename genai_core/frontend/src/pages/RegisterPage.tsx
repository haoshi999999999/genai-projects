import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { registerUser } from '../services/api'
import './RegisterPage.css'

type Language = 'en' | 'zh-CN' | 'zh-TW' | 'de' | 'fr' | 'ru'

const translations = {
  en: {
    title: 'Create Account',
    subtitle: 'Create your GenAI Assistant account.',
    firstName: 'First Name',
    lastName: 'Last Name',
    email: 'Email',
    password: 'Password',
    confirmPassword: 'Confirm Password',
    firstNamePlaceholder: 'Enter your first name',
    lastNamePlaceholder: 'Enter your last name',
    emailPlaceholder: 'Enter your email',
    passwordPlaceholder: 'Enter your password',
    confirmPasswordPlaceholder: 'Confirm your password',
    register: 'Register',
    registering: 'Creating account...',
    backToLogin: 'Already have an account? Login',
    nameRequired: 'Please enter your first and last name.',
    nameTooLong: 'Names cannot exceed 100 characters.',
    passwordMismatch: 'Passwords do not match.',
    passwordRequired: 'Please enter a password.',
    success: 'Account created successfully. Please log in.',
  },

  'zh-CN': {
    title: '创建账户',
    subtitle: '创建您的 GenAI Assistant 账户。',
    firstName: '名字',
    lastName: '姓氏',
    email: '邮箱',
    password: '密码',
    confirmPassword: '确认密码',
    firstNamePlaceholder: '请输入名字',
    lastNamePlaceholder: '请输入姓氏',
    emailPlaceholder: '请输入邮箱',
    passwordPlaceholder: '请输入密码',
    confirmPasswordPlaceholder: '请再次输入密码',
    register: '注册',
    registering: '正在创建账户...',
    backToLogin: '已有账户？登录',
    nameRequired: '请输入名字和姓氏。',
    nameTooLong: '名字和姓氏不能超过100个字符。',
    passwordMismatch: '两次输入的密码不一致。',
    passwordRequired: '请输入密码。',
    success: '账户创建成功，请登录。',
  },

  'zh-TW': {
    title: '建立帳戶',
    subtitle: '建立您的 GenAI Assistant 帳戶。',
    firstName: '名字',
    lastName: '姓氏',
    email: '電子郵件',
    password: '密碼',
    confirmPassword: '確認密碼',
    firstNamePlaceholder: '請輸入名字',
    lastNamePlaceholder: '請輸入姓氏',
    emailPlaceholder: '請輸入電子郵件',
    passwordPlaceholder: '請輸入密碼',
    confirmPasswordPlaceholder: '請再次輸入密碼',
    register: '註冊',
    registering: '正在建立帳戶...',
    backToLogin: '已有帳戶？登入',
    nameRequired: '請輸入名字和姓氏。',
    nameTooLong: '名字和姓氏不能超過100個字元。',
    passwordMismatch: '兩次輸入的密碼不一致。',
    passwordRequired: '請輸入密碼。',
    success: '帳戶建立成功，請登入。',
  },

  de: {
    title: 'Konto erstellen',
    subtitle: 'Erstellen Sie Ihr GenAI Assistant-Konto.',
    firstName: 'Vorname',
    lastName: 'Nachname',
    email: 'E-Mail',
    password: 'Passwort',
    confirmPassword: 'Passwort bestätigen',
    firstNamePlaceholder: 'Vorname eingeben',
    lastNamePlaceholder: 'Nachname eingeben',
    emailPlaceholder: 'E-Mail eingeben',
    passwordPlaceholder: 'Passwort eingeben',
    confirmPasswordPlaceholder: 'Passwort erneut eingeben',
    register: 'Registrieren',
    registering: 'Konto wird erstellt...',
    backToLogin: 'Bereits ein Konto? Anmelden',
    nameRequired: 'Bitte geben Sie Ihren Vor- und Nachnamen ein.',
    nameTooLong: 'Namen dürfen maximal 100 Zeichen lang sein.',
    passwordMismatch: 'Die Passwörter stimmen nicht überein.',
    passwordRequired: 'Bitte geben Sie ein Passwort ein.',
    success: 'Konto erfolgreich erstellt. Bitte anmelden.',
  },

  fr: {
    title: 'Créer un compte',
    subtitle: 'Créez votre compte GenAI Assistant.',
    firstName: 'Prénom',
    lastName: 'Nom',
    email: 'E-mail',
    password: 'Mot de passe',
    confirmPassword: 'Confirmer le mot de passe',
    firstNamePlaceholder: 'Entrez votre prénom',
    lastNamePlaceholder: 'Entrez votre nom',
    emailPlaceholder: 'Entrez votre e-mail',
    passwordPlaceholder: 'Entrez votre mot de passe',
    confirmPasswordPlaceholder: 'Confirmez votre mot de passe',
    register: "S'inscrire",
    registering: 'Création du compte...',
    backToLogin: 'Déjà un compte ? Connexion',
    nameRequired: 'Veuillez saisir votre prénom et votre nom.',
    nameTooLong: 'Les noms ne peuvent pas dépasser 100 caractères.',
    passwordMismatch: 'Les mots de passe ne correspondent pas.',
    passwordRequired: 'Veuillez saisir un mot de passe.',
    success: 'Compte créé avec succès. Veuillez vous connecter.',
  },

  ru: {
    title: 'Создать аккаунт',
    subtitle: 'Создайте аккаунт GenAI Assistant.',
    firstName: 'Имя',
    lastName: 'Фамилия',
    email: 'Электронная почта',
    password: 'Пароль',
    confirmPassword: 'Подтвердите пароль',
    firstNamePlaceholder: 'Введите имя',
    lastNamePlaceholder: 'Введите фамилию',
    emailPlaceholder: 'Введите электронную почту',
    passwordPlaceholder: 'Введите пароль',
    confirmPasswordPlaceholder: 'Повторите пароль',
    register: 'Зарегистрироваться',
    registering: 'Создание аккаунта...',
    backToLogin: 'Уже есть аккаунт? Войти',
    nameRequired: 'Введите имя и фамилию.',
    nameTooLong: 'Имя и фамилия не должны превышать 100 символов.',
    passwordMismatch: 'Пароли не совпадают.',
    passwordRequired: 'Введите пароль.',
    success: 'Аккаунт успешно создан. Войдите в систему.',
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

function RegisterPage() {
  const navigate = useNavigate()

  const [firstName, setFirstName] = useState('')
  const [lastName, setLastName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')

  const [isLoading, setIsLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')

  const [language] = useState<Language>(() => {
    const savedLanguage = localStorage.getItem('language')

    if (
      savedLanguage &&
      supportedLanguages.includes(savedLanguage as Language)
    ) {
      return savedLanguage as Language
    }

    return 'en'
  })

  const t = translations[language]

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    setErrorMessage('')

    const normalizedFirstName = firstName.trim()
    const normalizedLastName = lastName.trim()

    if (!normalizedFirstName || !normalizedLastName) {
      setErrorMessage(t.nameRequired)
      return
    }

    if (
      normalizedFirstName.length > 100 ||
      normalizedLastName.length > 100
    ) {
      setErrorMessage(t.nameTooLong)
      return
    }

    if (!password.trim()) {
      setErrorMessage(t.passwordRequired)
      return
    }

    if (password !== confirmPassword) {
      setErrorMessage(t.passwordMismatch)
      return
    }

    setIsLoading(true)

    try {
      await registerUser({
        first_name: normalizedFirstName,
        last_name: normalizedLastName,
        email: email.trim(),
        password,
      })

      navigate('/', {
        replace: true,
        state: { registrationSuccess: t.success },
      })
    } catch (error) {
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
    <div className="register-container">
      <div className="register-box">
        <h1>{t.title}</h1>

        <p>{t.subtitle}</p>

        <form onSubmit={handleSubmit}>
          <div className="register-form-group">
            <label htmlFor="register-first-name">
              {t.firstName}
            </label>

            <input
              id="register-first-name"
              type="text"
              value={firstName}
              onChange={(event) => setFirstName(event.target.value)}
              placeholder={t.firstNamePlaceholder}
              autoComplete="given-name"
              maxLength={100}
              required
            />
          </div>

          <div className="register-form-group">
            <label htmlFor="register-last-name">
              {t.lastName}
            </label>

            <input
              id="register-last-name"
              type="text"
              value={lastName}
              onChange={(event) => setLastName(event.target.value)}
              placeholder={t.lastNamePlaceholder}
              autoComplete="family-name"
              maxLength={100}
              required
            />
          </div>

          <div className="register-form-group">
            <label htmlFor="register-email">{t.email}</label>

            <input
              id="register-email"
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder={t.emailPlaceholder}
              autoComplete="email"
              required
            />
          </div>

          <div className="register-form-group">
            <label htmlFor="register-password">{t.password}</label>

            <input
              id="register-password"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder={t.passwordPlaceholder}
              autoComplete="new-password"
              required
            />
          </div>

          <div className="register-form-group">
            <label htmlFor="confirm-password">
              {t.confirmPassword}
            </label>

            <input
              id="confirm-password"
              type="password"
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              placeholder={t.confirmPasswordPlaceholder}
              autoComplete="new-password"
              required
            />
          </div>

          {errorMessage && (
            <div className="register-error" role="alert">
              {errorMessage}
            </div>
          )}

          <button
            type="submit"
            className="register-button"
            disabled={isLoading}
          >
            {isLoading ? t.registering : t.register}
          </button>
        </form>

        <button
          type="button"
          className="register-login-link"
          onClick={() => navigate('/')}
        >
          {t.backToLogin}
        </button>
      </div>
    </div>
  )
}

export default RegisterPage