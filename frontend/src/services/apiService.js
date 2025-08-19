import axios from 'axios';

// API endpoints for different services
const API_ENDPOINTS = {
  main: process.env.REACT_APP_MAIN_API_URL || 'http://localhost:8000',
  chatbot: process.env.REACT_APP_CHATBOT_API_URL || 'http://localhost:8001', 
  orchestration: process.env.REACT_APP_ORCHESTRATION_API_URL || 'http://localhost:8002'
};

// Create axios instances for different APIs
const createApiInstance = (baseURL, timeout = 30000) => {
  const instance = axios.create({
    baseURL,
    timeout,
    headers: {
      'Content-Type': 'application/json',
    },
  });
  
  // Add request interceptor for logging
  instance.interceptors.request.use(
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
  instance.interceptors.response.use(
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
  
  return instance;
};

// Create API instances
const mainApi = createApiInstance(API_ENDPOINTS.main);
const chatbotApi = createApiInstance(API_ENDPOINTS.chatbot);
const orchestrationApi = createApiInstance(API_ENDPOINTS.orchestration);

// Legacy support - keep main API as default
const api = mainApi;



export const apiService = {
  // Health and Status
  async getHealth() {
    return await api.get('/health');
  },

  async getSystemStatus() {
    const response = await api.get('/status');
    return response.data;
  },

  // Chat endpoints (using dedicated chatbot API)
  async sendChatMessage(query, subreddits = null) {
    const response = await chatbotApi.post('/chat', {
      query,
      subreddits
    });
    return response;
  },

  async getChatSuggestions(subreddits = null) {
    const params = subreddits ? { subreddits } : {};
    const response = await chatbotApi.get('/suggestions', { params });
    return response;
  },

  async getChatTopics(subreddits = null) {
    const params = subreddits ? { subreddits } : {};
    const response = await chatbotApi.get('/insights/topics', { params });
    return response;
  },

  // Chatbot API health
  async getChatbotHealth() {
    return await chatbotApi.get('/health');
  },

  async getChatbotStatus() {
    return await chatbotApi.get('/status');
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

  // Orchestration API endpoints
  async getSchedulerStatus() {
    return await orchestrationApi.get('/scheduler/status');
  },

  async listScheduledJobs() {
    return await orchestrationApi.get('/scheduler/jobs');
  },

  async triggerJob(jobId, parameters = null) {
    return await orchestrationApi.post('/scheduler/jobs/trigger', {
      job_id: jobId,
      parameters
    });
  },

  async manageJob(jobId, action) {
    return await orchestrationApi.post('/scheduler/jobs/manage', {
      job_id: jobId,
      action
    });
  },

  async executeWorkflow(workflowType, query = null, subreddits = null, parameters = null) {
    return await orchestrationApi.post('/workflow/execute', {
      workflow_type: workflowType,
      query,
      subreddits,
      parameters
    });
  },

  async getWorkflowHistory(limit = 10) {
    return await orchestrationApi.get('/workflow/history', { params: { limit } });
  },

  async getSystemHealth() {
    return await orchestrationApi.get('/system/health');
  },

  async getMonitoringMetrics() {
    return await orchestrationApi.get('/monitoring/metrics');
  },

  async updateSchedulerConfig(config) {
    return await orchestrationApi.post('/scheduler/config', config);
  },

  // Orchestration API health
  async getOrchestrationHealth() {
    return await orchestrationApi.get('/health');
  },
};

// Utility functions for common operations
export const chatUtils = {
  async askQuestion(question, options = {}) {
    const {
      subreddits = null
    } = options;

    try {
      const response = await apiService.sendChatMessage(question, subreddits);
      
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

  async getSuggestions(subreddits = null) {
    try {
      const response = await apiService.getChatSuggestions(subreddits);
      return response.suggestions || [];
    } catch (error) {
      console.error('Error getting suggestions:', error);
      return [];
    }
  },

  async getTopics(subreddits = null) {
    try {
      const response = await apiService.getChatTopics(subreddits);
      return response.topics || [];
    } catch (error) {
      console.error('Error getting topics:', error);
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

export const orchestrationUtils = {
  async getSchedulerStatus() {
    try {
      const response = await apiService.getSchedulerStatus();
      return {
        success: true,
        data: response.data,
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

  async getJobs() {
    try {
      const response = await apiService.listScheduledJobs();
      return {
        success: true,
        data: response.data,
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

  async triggerJob(jobId) {
    try {
      const response = await apiService.triggerJob(jobId);
      return {
        success: true,
        data: response.data,
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

  async toggleJob(jobId, enable = true) {
    try {
      const action = enable ? 'enable' : 'disable';
      const response = await apiService.manageJob(jobId, action);
      return {
        success: true,
        data: response.data,
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

  async executeWorkflow(workflowType, options = {}) {
    const { query = null, subreddits = null, parameters = null } = options;
    
    try {
      const response = await apiService.executeWorkflow(workflowType, query, subreddits, parameters);
      return {
        success: true,
        data: response.data,
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

  async getSystemHealth() {
    try {
      const response = await apiService.getSystemHealth();
      return {
        success: true,
        data: response.data,
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

  async getMetrics() {
    try {
      const response = await apiService.getMonitoringMetrics();
      return {
        success: true,
        data: response.data,
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
