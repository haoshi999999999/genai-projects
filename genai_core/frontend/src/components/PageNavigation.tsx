import { useNavigate, useLocation } from 'react-router-dom'
import { getSavedLanguage } from '../i18n/language'
import { navigationTranslations } from '../i18n/navigationTranslations'
import './PageNavigation.css'

interface PageNavigationProps {
  logoutLabel?: string
}

function PageNavigation({ logoutLabel }: PageNavigationProps) {
  const navigate = useNavigate()
  const location = useLocation()
  const language = getSavedLanguage()
  const t = navigationTranslations[language]
  const translatedLogoutLabel = logoutLabel ?? t.logout

  const handleLogout = () => {
    sessionStorage.removeItem('access_token')
    navigate('/', { replace: true })
  }

  return (
    <nav className="page-navigation">
      <div className="navigation-brand">GenAI Assistant</div>

      <div className="navigation-links">
        <button
          type="button"
          className={
            location.pathname === '/chat'
              ? 'navigation-link active'
              : 'navigation-link'
          }
          onClick={() => navigate('/chat')}
        >
          {t.chat}
        </button>

        <button
          type="button"
          className={
            location.pathname === '/documents'
              ? 'navigation-link active'
              : 'navigation-link'
          }
          onClick={() => navigate('/documents')}
        >
          {t.documents}
        </button>

        <button
          type="button"
          className={
            location.pathname === '/settings'
              ? 'navigation-link active'
              : 'navigation-link'
          }
          onClick={() => navigate('/settings')}
        >
          {t.settings}
        </button>
      </div>

      <button
        type="button"
        className="logout-button"
        onClick={handleLogout}
      >
        {translatedLogoutLabel}
      </button>
    </nav>
  )
}

export default PageNavigation