import type { Language } from './language'

export const navigationTranslations = {
  en: {
    chat: 'Chat',
    documents: 'Documents',
    settings: 'Settings',
    logout: 'Logout',
  },

  'zh-CN': {
    chat: '聊天',
    documents: '文档',
    settings: '设置',
    logout: '退出登录',
  },

  'zh-TW': {
    chat: '聊天',
    documents: '文件',
    settings: '設定',
    logout: '登出',
  },

  de: {
    chat: 'Chat',
    documents: 'Dokumente',
    settings: 'Einstellungen',
    logout: 'Abmelden',
  },

  fr: {
    chat: 'Discussion',
    documents: 'Documents',
    settings: 'Paramètres',
    logout: 'Déconnexion',
  },

  ru: {
    chat: 'Чат',
    documents: 'Документы',
    settings: 'Настройки',
    logout: 'Выйти',
  },
} satisfies Record<Language, Record<string, string>>