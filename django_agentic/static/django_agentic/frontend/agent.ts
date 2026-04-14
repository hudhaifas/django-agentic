import api from './client';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface InterruptAction {
  name: string;
  args: Record<string, unknown>;
  id: string;
  description: string;
}

export interface InterruptData {
  message: string;
  actions: InterruptAction[];
  next_nodes: string[];
}

export interface ChatResponse {
  success: boolean;
  message: string;
  usage?: Record<string, number>;
  interrupt?: InterruptData;
  error?: string;
}

export const agentApi = {
  chat: (message: string, entityClass: string, entityId: string) =>
    api.post<ChatResponse>('/agentic/agent/chat', {
      message,
      context: { entity_class: entityClass, entity_id: entityId },
    }),

  resume: (entityClass: string, entityId: string, approved: boolean) =>
    api.post<ChatResponse>('/agentic/agent/resume', {
      approved,
      context: { entity_class: entityClass, entity_id: entityId },
    }),

  history: (entityClass: string, entityId: string) =>
    api.get<{ history: ChatMessage[] }>('/agentic/agent/history', {
      params: { entity_class: entityClass, entity_id: entityId },
    }),
};
