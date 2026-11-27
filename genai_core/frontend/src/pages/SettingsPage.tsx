import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import PageNavigation from '../components/PageNavigation'

import {
  getMyAccount,
  updateMyAccount,
  deleteMyAccount,
} from '../services/api'

import { getSavedLanguage } from '../i18n/language'
import { settingsTranslations } from '../i18n/settingsTranslations'

import './SettingsPage.css'

function SettingsPage() {
  const navigate = useNavigate()

  // Load saved language preference
  const language = getSavedLanguage()
  const t = settingsTranslations[language]

  // Account information
  const [email, setEmail] = useState('')
  const [firstName, setFirstName] = useState('')
  const [lastName, setLastName] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const [isSavingProfile, setIsSavingProfile] = useState(false)

  // Password form
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [isUpdating, setIsUpdating] = useState(false)

  // Account deletion
  const [isDeleting, setIsDeleting] = useState(false)

  // Feedback
  const [errorMessage, setErrorMessage] = useState('')
  const [successMessage, setSuccessMessage] = useState('')

  // Load authenticated user's account information
  useEffect(() => {
    const loadAccount = async () => {
      try {
        setIsLoading(true)
        setErrorMessage('')

        const result = await getMyAccount()

        setEmail(result.email)
        setFirstName(result.first_name ?? '')
        setLastName(result.last_name ?? '')
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

    void loadAccount()
  }, [])

  // Update profile information
  const handleProfileUpdate = async (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault()

    if (isSavingProfile || isUpdating || isDeleting) {
      return
    }

    setErrorMessage('')
    setSuccessMessage('')

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

    try {
      setIsSavingProfile(true)

      const result = await updateMyAccount({
        first_name: normalizedFirstName,
        last_name: normalizedLastName,
      })

      setFirstName(result.first_name ?? '')
      setLastName(result.last_name ?? '')

      setSuccessMessage(t.profileUpdated)
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : t.profileUpdateError
      )
    } finally {
      setIsSavingProfile(false)
    }
  }

  // Update password
  const handlePasswordUpdate = async (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault()

    if (isUpdating || isDeleting || isSavingProfile) {
      return
    }

    setErrorMessage('')
    setSuccessMessage('')

    if (!newPassword.trim()) {
      setErrorMessage(t.passwordRequired)
      return
    }

    if (newPassword !== confirmPassword) {
      setErrorMessage(t.passwordMismatch)
      return
    }

    try {
      setIsUpdating(true)

      await updateMyAccount({
        password: newPassword,
      })

      setNewPassword('')
      setConfirmPassword('')

      setSuccessMessage(t.passwordUpdated)
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : t.passwordUpdateError
      )
    } finally {
      setIsUpdating(false)
    }
  }

  // Permanently delete account
  const handleAccountDeletion = async () => {
    if (isDeleting || isUpdating || isSavingProfile) {
      return
    }

    const confirmed = window.confirm(t.deleteConfirm)

    if (!confirmed) {
      return
    }

    setErrorMessage('')
    setSuccessMessage('')

    try {
      setIsDeleting(true)

      await deleteMyAccount()

      // Clear authentication only after backend confirms deletion
      sessionStorage.removeItem('access_token')

      // Return to login page
      navigate('/', { replace: true })
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : t.deleteError
      )
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <>
      <PageNavigation />

      <main className="settings-container">
        <header className="settings-header">
          <h1>{t.title}</h1>
          <p>{t.description}</p>
        </header>

        {errorMessage && (
          <div className="settings-error" role="alert">
            {errorMessage}
          </div>
        )}

        {successMessage && (
          <div className="settings-success" role="status">
            {successMessage}
          </div>
        )}

        {/* Account Information */}
        <section className="settings-section">
          <h2>{t.accountSection}</h2>

          {isLoading ? (
            <p>{t.loading}</p>
          ) : (
            <form
              className="settings-account-info"
              onSubmit={handleProfileUpdate}
            >
              <div className="settings-form-group">
                <label htmlFor="account-email">
                  {t.email}
                </label>

                <input
                  id="account-email"
                  type="email"
                  value={email}
                  readOnly
                />
              </div>

              <div className="settings-form-group">
                <label htmlFor="first-name">
                  {t.firstName}
                </label>

                <input
                  id="first-name"
                  type="text"
                  value={firstName}
                  onChange={(event) =>
                    setFirstName(event.target.value)
                  }
                  maxLength={100}
                  required
                  disabled={
                    isSavingProfile || isUpdating || isDeleting
                  }
                />
              </div>

              <div className="settings-form-group">
                <label htmlFor="last-name">
                  {t.lastName}
                </label>

                <input
                  id="last-name"
                  type="text"
                  value={lastName}
                  onChange={(event) =>
                    setLastName(event.target.value)
                  }
                  maxLength={100}
                  required
                  disabled={
                    isSavingProfile || isUpdating || isDeleting
                  }
                />
              </div>

              <button
                type="submit"
                className="settings-primary-button"
                disabled={
                  isSavingProfile || isUpdating || isDeleting
                }
              >
                {isSavingProfile
                  ? t.savingProfile
                  : t.saveChanges}
              </button>
            </form>
          )}
        </section>

        {/* Change Password */}
        <section className="settings-section">
          <h2>{t.passwordSection}</h2>

          <p className="settings-description">
            {t.passwordDescription}
          </p>

          <form onSubmit={handlePasswordUpdate}>
            <div className="settings-form-group">
              <label htmlFor="new-password">
                {t.newPassword}
              </label>

              <input
                id="new-password"
                type="password"
                value={newPassword}
                onChange={(event) =>
                  setNewPassword(event.target.value)
                }
                autoComplete="new-password"
                required
                disabled={
                  isUpdating || isDeleting || isSavingProfile
                }
              />
            </div>

            <div className="settings-form-group">
              <label htmlFor="confirm-password">
                {t.confirmPassword}
              </label>

              <input
                id="confirm-password"
                type="password"
                value={confirmPassword}
                onChange={(event) =>
                  setConfirmPassword(event.target.value)
                }
                autoComplete="new-password"
                required
                disabled={
                  isUpdating || isDeleting || isSavingProfile
                }
              />
            </div>

            <button
              type="submit"
              className="settings-primary-button"
              disabled={
                isUpdating || isDeleting || isSavingProfile
              }
            >
              {isUpdating ? t.updating : t.updatePassword}
            </button>
          </form>
        </section>

        {/* Danger Zone */}
        <section className="settings-section settings-danger-section">
          <h2>{t.dangerSection}</h2>

          <p className="settings-description">
            {t.dangerDescription}
          </p>

          <button
            type="button"
            className="settings-danger-button"
            onClick={handleAccountDeletion}
            disabled={
              isDeleting || isUpdating || isSavingProfile
            }
          >
            {isDeleting ? t.deleting : t.deleteAccount}
          </button>
        </section>
      </main>
    </>
  )
}

export default SettingsPage