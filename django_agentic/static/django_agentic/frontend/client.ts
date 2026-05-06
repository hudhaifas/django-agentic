/**
 * Minimal axios client for django_agentic API.
 *
 * Copy this into your frontend and adjust the baseURL and CSRF
 * configuration to match your project setup.
 */
import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  xsrfCookieName: 'csrftoken',
  xsrfHeaderName: 'X-CSRFToken',
  withCredentials: true,
});

export default api;
