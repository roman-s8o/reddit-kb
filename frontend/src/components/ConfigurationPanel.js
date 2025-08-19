import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Chip,
  Alert,
  CircularProgress,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Paper,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Storage as StorageIcon,
  Psychology as PsychologyIcon,
  Chat as ChatIcon,
  CloudDownload as CloudDownloadIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { apiService, workflowUtils, orchestrationUtils } from '../services/apiService';

const ConfigurationPanel = ({ systemStatus, onStatusUpdate }) => {
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [testingConnection, setTestingConnection] = useState(false);
  const [orchestrationHealth, setOrchestrationHealth] = useState(null);
  const [systemHealth, setSystemHealth] = useState(null);

  useEffect(() => {
    loadConfiguration();
    loadHealthStatus();
  }, []);

  const loadConfiguration = async () => {
    try {
      setLoading(true);
      const configData = await apiService.getConfiguration();
      setConfig(configData);
      setError(null);
    } catch (err) {
      console.error('Error loading configuration:', err);
      setError('Failed to load configuration');
    } finally {
      setLoading(false);
    }
  };

  const loadHealthStatus = async () => {
    try {
      // Test orchestration API
      const orchestrationResult = await orchestrationUtils.getSystemHealth();
      if (orchestrationResult.success) {
        setOrchestrationHealth(orchestrationResult.data);
        setSystemHealth(orchestrationResult.data.system_status);
      }
    } catch (err) {
      console.error('Error loading health status:', err);
    }
  };

  const testConnection = async () => {
    try {
      setTestingConnection(true);
      await apiService.getHealth();
      await onStatusUpdate();
      await loadHealthStatus();
      setError(null);
    } catch (err) {
      setError('Connection test failed: ' + err.message);
    } finally {
      setTestingConnection(false);
    }
  };

  const getStatusIcon = (status) => {
    if (status === 'ready') {
      return <CheckCircleIcon color="success" />;
    } else if (status === 'error') {
      return <ErrorIcon color="error" />;
    } else {
      return <InfoIcon color="info" />;
    }
  };

  const getStatusColor = (status) => {
    if (status === 'ready') return 'success';
    if (status === 'error') return 'error';
    return 'info';
  };

  const formatBytes = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading configuration...
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          System Configuration
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => {
              loadConfiguration();
              loadHealthStatus();
              onStatusUpdate();
            }}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={testingConnection ? <CircularProgress size={20} /> : <CheckCircleIcon />}
            onClick={testConnection}
            disabled={testingConnection}
          >
            Test Connection
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* API Status */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                API Services Status
              </Typography>
              
              <Grid container spacing={2}>
                {/* Main API */}
                <Grid item xs={12} md={4}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <StorageIcon sx={{ mr: 1 }} />
                      <Typography variant="subtitle1">Main API</Typography>
                      {systemStatus ? (
                        <Chip 
                          label="Connected" 
                          color="success" 
                          size="small" 
                          sx={{ ml: 'auto' }} 
                        />
                      ) : (
                        <Chip 
                          label="Disconnected" 
                          color="error" 
                          size="small" 
                          sx={{ ml: 'auto' }} 
                        />
                      )}
                    </Box>
                    <Typography variant="body2" color="textSecondary">
                      Port: 8000 • Data Collection & Insights
                    </Typography>
                  </Paper>
                </Grid>

                {/* Chatbot API */}
                <Grid item xs={12} md={4}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <ChatIcon sx={{ mr: 1 }} />
                      <Typography variant="subtitle1">Chatbot API</Typography>
                      <Chip 
                        label="Available" 
                        color="info" 
                        size="small" 
                        sx={{ ml: 'auto' }} 
                      />
                    </Box>
                    <Typography variant="body2" color="textSecondary">
                      Port: 8001 • Chat & Question Answering
                    </Typography>
                  </Paper>
                </Grid>

                {/* Orchestration API */}
                <Grid item xs={12} md={4}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <SettingsIcon sx={{ mr: 1 }} />
                      <Typography variant="subtitle1">Orchestration API</Typography>
                      {orchestrationHealth ? (
                        <Chip 
                          label="Connected" 
                          color="success" 
                          size="small" 
                          sx={{ ml: 'auto' }} 
                        />
                      ) : (
                        <Chip 
                          label="Disconnected" 
                          color="warning" 
                          size="small" 
                          sx={{ ml: 'auto' }} 
                        />
                      )}
                    </Box>
                    <Typography variant="body2" color="textSecondary">
                      Port: 8002 • Workflow Management & Scheduling
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>

              {/* Health Checks */}
              {orchestrationHealth && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    System Health Checks
                  </Typography>
                  <Grid container spacing={1}>
                    {Object.entries(orchestrationHealth.health_checks || {}).map(([check, status]) => (
                      <Grid item xs={12} sm={6} md={3} key={check}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          {status ? (
                            <CheckCircleIcon color="success" sx={{ mr: 1, fontSize: 16 }} />
                          ) : (
                            <ErrorIcon color="error" sx={{ mr: 1, fontSize: 16 }} />
                          )}
                          <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                            {check.replace(/_/g, ' ')}
                          </Typography>
                        </Box>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* System Status */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Status
              </Typography>
              
              {systemStatus ? (
                <Grid container spacing={2}>
                  {/* Collector Status */}
                  <Grid item xs={12} md={6} lg={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <CloudDownloadIcon sx={{ fontSize: 40, mb: 1, color: 'primary.main' }} />
                      <Typography variant="subtitle1">Data Collector</Typography>
                      <Chip
                        icon={getStatusIcon(systemStatus.collector?.status)}
                        label={systemStatus.collector?.status || 'Unknown'}
                        color={getStatusColor(systemStatus.collector?.status)}
                        size="small"
                        sx={{ mt: 1 }}
                      />
                      <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                        Reddit API: {systemStatus.collector?.reddit_client ? 'Connected' : 'Not Connected'}
                      </Typography>
                    </Paper>
                  </Grid>

                  {/* Preprocessor Status */}
                  <Grid item xs={12} md={6} lg={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <StorageIcon sx={{ fontSize: 40, mb: 1, color: 'secondary.main' }} />
                      <Typography variant="subtitle1">Preprocessor</Typography>
                      <Chip
                        icon={getStatusIcon(systemStatus.preprocessor?.status)}
                        label={systemStatus.preprocessor?.status || 'Unknown'}
                        color={getStatusColor(systemStatus.preprocessor?.status)}
                        size="small"
                        sx={{ mt: 1 }}
                      />
                      <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                        Vector DB: {systemStatus.preprocessor?.chroma_connected ? 'Connected' : 'Not Connected'}
                      </Typography>
                    </Paper>
                  </Grid>

                  {/* Insight Agent Status */}
                  <Grid item xs={12} md={6} lg={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <PsychologyIcon sx={{ fontSize: 40, mb: 1, color: 'success.main' }} />
                      <Typography variant="subtitle1">Insight Agent</Typography>
                      <Chip
                        icon={getStatusIcon(systemStatus.insight_agent?.status)}
                        label={systemStatus.insight_agent?.status || 'Unknown'}
                        color={getStatusColor(systemStatus.insight_agent?.status)}
                        size="small"
                        sx={{ mt: 1 }}
                      />
                      <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                        Insights: {systemStatus.insight_agent?.latest_insights || 0}
                      </Typography>
                    </Paper>
                  </Grid>

                  {/* Chatbot Status */}
                  <Grid item xs={12} md={6} lg={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <ChatIcon sx={{ fontSize: 40, mb: 1, color: 'warning.main' }} />
                      <Typography variant="subtitle1">Chatbot</Typography>
                      <Chip
                        icon={getStatusIcon(systemStatus.chatbot?.status)}
                        label={systemStatus.chatbot?.status || 'Unknown'}
                        color={getStatusColor(systemStatus.chatbot?.status)}
                        size="small"
                        sx={{ mt: 1 }}
                      />
                      <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                        Documents: {systemStatus.chatbot?.knowledge_base?.total_documents || 0}
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
              ) : (
                <Alert severity="warning">
                  System status unavailable. Check connection to backend.
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Configuration Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Data Collection Settings
              </Typography>
              
              {config && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Monitored Subreddits:
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                    {config.subreddits?.map((subreddit) => (
                      <Chip 
                        key={subreddit} 
                        label={`r/${subreddit}`} 
                        size="small" 
                        color="primary"
                        variant="outlined"
                      />
                    ))}
                  </Box>

                  <TextField
                    fullWidth
                    label="Max Posts per Subreddit"
                    value={config.max_posts_per_subreddit || ''}
                    disabled
                    size="small"
                    sx={{ mb: 2 }}
                  />

                  <TextField
                    fullWidth
                    label="Collection Interval (hours)"
                    value={config.collection_interval_hours || ''}
                    disabled
                    size="small"
                    sx={{ mb: 2 }}
                  />

                  <TextField
                    fullWidth
                    label="Insight Generation Interval (hours)"
                    value={config.insight_generation_interval_hours || ''}
                    disabled
                    size="small"
                  />
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* AI Model Configuration */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                AI Model Configuration
              </Typography>
              
              {config && (
                <Box>
                  <TextField
                    fullWidth
                    label="Ollama LLM Model"
                    value={config.ollama_model || ''}
                    disabled
                    size="small"
                    sx={{ mb: 2 }}
                  />

                  <TextField
                    fullWidth
                    label="Ollama Embedding Model"
                    value={config.ollama_embedding_model || ''}
                    disabled
                    size="small"
                    sx={{ mb: 2 }}
                  />

                  <Alert severity="info" sx={{ mt: 2 }}>
                    Model configuration is currently read-only. Modify settings in the backend configuration file.
                  </Alert>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Database Statistics */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Database Statistics
              </Typography>
              
              {systemStatus?.preprocessor?.collection_stats && (
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="h4" color="primary.main">
                        {systemStatus.preprocessor.collection_stats.total_documents || 0}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Total Documents in Vector DB
                      </Typography>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="h4" color="secondary.main">
                        {Object.keys(systemStatus.preprocessor.collection_stats.subreddit_distribution || {}).length}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Subreddits in Database
                      </Typography>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="h4" color="success.main">
                        {systemStatus.insight_agent?.latest_insights || 0}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Generated Insights
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
              )}

              {systemStatus?.preprocessor?.collection_stats?.subreddit_distribution && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Document Distribution by Subreddit:
                  </Typography>
                  <List dense>
                    {Object.entries(systemStatus.preprocessor.collection_stats.subreddit_distribution)
                      .sort(([,a], [,b]) => b - a)
                      .slice(0, 10)
                      .map(([subreddit, count]) => (
                        <ListItem key={subreddit}>
                          <ListItemText
                            primary={`r/${subreddit}`}
                            secondary={`${count} documents`}
                          />
                        </ListItem>
                      ))}
                  </List>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<CloudDownloadIcon />}
                    onClick={() => workflowUtils.collectData()}
                    size="large"
                  >
                    Collect New Data
                  </Button>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<PsychologyIcon />}
                    onClick={() => workflowUtils.runFullWorkflow()}
                    size="large"
                  >
                    Generate Insights
                  </Button>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={<SettingsIcon />}
                    onClick={() => workflowUtils.runFullWorkflow()}
                    size="large"
                  >
                    Run Full Workflow
                  </Button>
                </Grid>
              </Grid>

              <Alert severity="info" sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Workflow Information:
                </Typography>
                <Typography variant="body2">
                  • <strong>Collect New Data:</strong> Fetches latest posts from configured subreddits<br/>
                  • <strong>Generate Insights:</strong> Analyzes existing data to extract topics and sentiment<br/>
                  • <strong>Run Full Workflow:</strong> Complete pipeline from data collection to insight generation
                </Typography>
              </Alert>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ConfigurationPanel;
