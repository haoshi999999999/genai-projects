import type { Language } from './language'

export const navigationTranslations = {
  en: {
    chat: 'Chat',
    documents: 'Documents',
    logout: 'Logout',
  },

  'zh-CN': {
    chat: '聊天',
    documents: '文档',
    logout: '退出登录',
  },

  'zh-TW': {
    chat: '聊天',
    documents: '文件',
    logout: '登出',
  },

  de: {
    chat: 'Chat',
    documents: 'Dokumente',
    logout: 'Abmelden',
  },

  fr: {
    chat: 'Discussion',
    documents: 'Documents',
    logout: 'Déconnexion',
  },

  ru: {
    chat: 'Чат',
    documents: 'Документы',
    logout: 'Выйти',
  },
} satisfies Record<Language, Record<string, string>>