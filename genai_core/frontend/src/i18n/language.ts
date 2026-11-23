export type Language =
  | 'en'
  | 'zh-CN'
  | 'zh-TW'
  | 'de'
  | 'fr'
  | 'ru'

export const supportedLanguages: Language[] = [
  'en',
  'zh-CN',
  'zh-TW',
  'de',
  'fr',
  'ru',
]

export function getSavedLanguage(): Language {
  const savedLanguage = localStorage.getItem('language')

  if (
    savedLanguage &&
    supportedLanguages.includes(savedLanguage as Language)
  ) {
    return savedLanguage as Language
  }

  return 'en'
}

export function saveLanguage(language: Language): void {
  localStorage.setItem('language', language)
}