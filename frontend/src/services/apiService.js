import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    console.error('API Response Error:', error);
    
    if (error.response) {
      // Server responded with error status
      throw new Error(error.response.data?.message || error.response.data?.detail || 'Server error');
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('No response from server. Please check if the API is running.');
    } else {
      // Something else happened
      throw new Error(error.message || 'Unknown error occurred');
    }
  }
);

export const apiService = {
  // Health and Status
  async getHealth() {
    return await api.get('/health');
  },

  async getSystemStatus() {
    const response = await api.get('/status');
    return response.data;
  },

  // Chat endpoints
  async sendChatMessage(query, subreddits = null, includeInsights = true, maxResults = 10) {
    const response = await api.post('/chat', {
      query,
      subreddits,
      include_insights: includeInsights,
      max_results: maxResults
    });
    return response.data;
  },

  async getChatSuggestions(query = null) {
    const params = query ? { query } : {};
    const response = await api.get('/chat/suggestions', { params });
    return response.data;
  },

  // Data Collection endpoints
  async startDataCollection(subreddits = null, maxPosts = null, timeFilter = 'day', sortBy = 'hot') {
    const response = await api.post('/collect', {
      subreddits,
      max_posts_per_subreddit: maxPosts,
      time_filter: timeFilter,
      sort_by: sortBy
    });
    return response.data;
  },

  async runDataCollectionSync(subreddits = null, maxPosts = null, timeFilter = 'day', sortBy = 'hot') {
    const response = await api.post('/collect/sync', {
      subreddits,
      max_posts_per_subreddit: maxPosts,
      time_filter: timeFilter,
      sort_by: sortBy
    });
    return response.data;
  },

  // Insights endpoints
  async generateInsights(subreddits = null, clusteringMethod = 'kmeans', nClusters = null) {
    const response = await api.post('/insights/generate', {
      subreddits,
      clustering_method: clusteringMethod,
      n_clusters: nClusters
    });
    return response.data;
  },

  async getLatestInsights(limit = 10) {
    const response = await api.get('/insights/latest', { params: { limit } });
    return response.data;
  },

  async getInsightsDashboard() {
    const response = await api.get('/insights/dashboard');
    return response.data;
  },

  // Topic Analysis
  async getTopicSummary(topic, subreddits = null, maxDocuments = 20) {
    const response = await api.post('/topic/summary', {
      topic,
      subreddits,
      max_documents: maxDocuments
    });
    return response.data;
  },

  // Workflow Management
  async runBatchWorkflow(subreddits = null, background = true) {
    const params = { background };
    if (subreddits) {
      params.subreddits = subreddits;
    }
    const response = await api.post('/workflow/batch', null, { params });
    return response.data;
  },

  // Configuration
  async getConfiguration() {
    const response = await api.get('/config');
    return response.data;
  },
};

// Utility functions for common operations
export const chatUtils = {
  async askQuestion(question, options = {}) {
    const {
      subreddits = null,
      includeInsights = true,
      maxResults = 10
    } = options;

    try {
      const response = await apiService.sendChatMessage(
        question,
        subreddits,
        includeInsights,
        maxResults
      );
      
      return {
        success: true,
        data: response,
        error: null
      };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error.message
      };
    }
  },

  async getSuggestions(query = null) {
    try {
      const response = await apiService.getChatSuggestions(query);
      return response.suggestions || [];
    } catch (error) {
      console.error('Error getting suggestions:', error);
      return [];
    }
  }
};

export const workflowUtils = {
  async runFullWorkflow(subreddits = null, background = true) {
    try {
      const response = await apiService.runBatchWorkflow(subreddits, background);
      return {
        success: true,
        data: response,
        error: null
      };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error.message
      };
    }
  },

  async collectData(subreddits = null, options = {}) {
    const {
      maxPosts = null,
      timeFilter = 'day',
      sortBy = 'hot',
      sync = false
    } = options;

    try {
      const response = sync 
        ? await apiService.runDataCollectionSync(subreddits, maxPosts, timeFilter, sortBy)
        : await apiService.startDataCollection(subreddits, maxPosts, timeFilter, sortBy);
      
      return {
        success: true,
        data: response,
        error: null
      };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error.message
      };
    }
  }
};

export const insightUtils = {
  async getLatest(limit = 10) {
    try {
      const response = await apiService.getLatestInsights(limit);
      return response.insights || [];
    } catch (error) {
      console.error('Error getting latest insights:', error);
      return [];
    }
  },

  async getDashboardData() {
    try {
      const response = await apiService.getInsightsDashboard();
      return response;
    } catch (error) {
      console.error('Error getting dashboard data:', error);
      return null;
    }
  },

  async generate(subreddits = null, options = {}) {
    const {
      clusteringMethod = 'kmeans',
      nClusters = null
    } = options;

    try {
      const response = await apiService.generateInsights(subreddits, clusteringMethod, nClusters);
      return {
        success: true,
        data: response,
        error: null
      };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error.message
      };
    }
  }
};

// Export default api instance for custom requests
export default api;
