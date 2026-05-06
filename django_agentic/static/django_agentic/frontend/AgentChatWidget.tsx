/**
 * Generic agent chat widget for django_agentic.
 *
 * Copy this file into your frontend and customize as needed.
 * Requires: react, antd, react-markdown, and the agent.ts API client.
 *
 * Usage:
 *   <AgentChatWidget entityClass="myapp.MyModel" entityId={id} />
 */
import { useState, useRef, useEffect } from 'react';
import { Input, Button, Spin, Typography, Card, Space, Tag } from 'antd';
import {
  SendOutlined, RobotOutlined, UserOutlined,
  CheckOutlined, CloseOutlined, ToolOutlined,
} from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import { agentApi } from './agent';
import type { ChatMessage, InterruptData, ChatResponse } from './agent';

const { Text } = Typography;

interface Props {
  entityClass: string;
  entityId: string;
  placeholder?: string;
  accentColor?: string;
}

export default function AgentChatWidget({
  entityClass,
  entityId,
  placeholder = 'Ask a question...',
  accentColor = '#1677ff',
}: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [interrupt, setInterrupt] = useState<InterruptData | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    setHistoryLoading(true);
    agentApi.history(entityClass, entityId)
      .then(res => {
        if (!cancelled && res.data.history?.length) setMessages(res.data.history);
      })
      .catch(() => {})
      .finally(() => { if (!cancelled) setHistoryLoading(false); });
    return () => { cancelled = true; };
  }, [entityClass, entityId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading, interrupt]);

  const handleResponse = (data: ChatResponse) => {
    if (data.interrupt) {
      setInterrupt(data.interrupt);
    } else {
      setInterrupt(null);
      if (data.message) {
        setMessages(prev => [...prev, { role: 'assistant', content: data.message }]);
      }
    }
  };

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput('');
    setInterrupt(null);
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setLoading(true);
    try {
      const res = await agentApi.chat(text, entityClass, entityId);
      handleResponse(res.data);
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Something went wrong.' }]);
    } finally {
      setLoading(false);
    }
  };

  const handleApproval = async (approved: boolean) => {
    setLoading(true);
    setInterrupt(null);
    try {
      const res = await agentApi.resume(entityClass, entityId, approved);
      handleResponse(res.data);
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Something went wrong.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 300 }}>
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px 0' }}>
        {historyLoading && (
          <div style={{ textAlign: 'center', padding: 40 }}><Spin size="small" /></div>
        )}
        {!historyLoading && messages.length === 0 && !loading && (
          <div style={{ textAlign: 'center', padding: 40, color: '#999' }}>
            <RobotOutlined style={{ fontSize: 32, marginBottom: 8 }} />
            <div>{placeholder}</div>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} style={{
            display: 'flex', gap: 8, marginBottom: 12,
            flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
          }}>
            <div style={{
              width: 28, height: 28, borderRadius: '50%', display: 'flex',
              alignItems: 'center', justifyContent: 'center', flexShrink: 0,
              background: msg.role === 'user' ? accentColor : '#f0f0f0',
              color: msg.role === 'user' ? '#fff' : '#666', fontSize: 14,
            }}>
              {msg.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
            </div>
            <div style={{
              maxWidth: '80%', padding: '8px 12px', borderRadius: 8,
              background: msg.role === 'user' ? accentColor : '#f5f5f5',
              color: msg.role === 'user' ? '#fff' : '#333',
            }}>
              {msg.role === 'assistant'
                ? <ReactMarkdown>{msg.content}</ReactMarkdown>
                : <Text style={{ color: 'inherit' }}>{msg.content}</Text>}
            </div>
          </div>
        ))}
        {interrupt && (
          <Card size="small" style={{ margin: '8px 0', borderColor: '#faad14' }}>
            <Space direction="vertical" size={8} style={{ width: '100%' }}>
              <Text strong>{interrupt.message}</Text>
              {interrupt.actions.map((action, i) => (
                <div key={i} style={{ padding: '4px 8px', background: '#fafafa', borderRadius: 4 }}>
                  <Tag icon={<ToolOutlined />} color="warning">{action.name}</Tag>
                  <Text type="secondary" style={{ fontSize: 13 }}>{action.description}</Text>
                </div>
              ))}
              <Space>
                <Button type="primary" icon={<CheckOutlined />} size="small"
                  onClick={() => handleApproval(true)} loading={loading}>Approve</Button>
                <Button danger icon={<CloseOutlined />} size="small"
                  onClick={() => handleApproval(false)} loading={loading}>Reject</Button>
              </Space>
            </Space>
          </Card>
        )}
        {loading && !interrupt && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            <div style={{
              width: 28, height: 28, borderRadius: '50%', display: 'flex',
              alignItems: 'center', justifyContent: 'center',
              background: '#f0f0f0', color: '#666', fontSize: 14,
            }}><RobotOutlined /></div>
            <Spin size="small" style={{ marginTop: 6 }} />
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div style={{ display: 'flex', gap: 8, paddingTop: 8, borderTop: '1px solid #f0f0f0' }}>
        <Input.TextArea
          value={input}
          onChange={e => setInput(e.target.value)}
          onPressEnter={e => { if (!e.shiftKey) { e.preventDefault(); send(); } }}
          placeholder={placeholder}
          autoSize={{ minRows: 1, maxRows: 4 }}
          disabled={loading || !!interrupt}
        />
        <Button type="primary" icon={<SendOutlined />} onClick={send}
          loading={loading} disabled={!!interrupt} style={{ alignSelf: 'flex-end' }} />
      </div>
    </div>
  );
}
