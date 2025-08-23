import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Tabs,
  Tab,
  Box,
  Paper,
  Alert,
  CircularProgress,
  Fab
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import ChatIcon from '@mui/icons-material/Chat';
import InsightsIcon from '@mui/icons-material/Insights';
import SettingsIcon from '@mui/icons-material/Settings';
import RefreshIcon from '@mui/icons-material/Refresh';

import ChatInterface from './components/ChatInterface';
import InsightsDashboard from './components/InsightsDashboard';
import ConfigurationPanel from './components/ConfigurationPanel';
import { apiService } from './services/apiService';

// Create Material-UI theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
});

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function App() {
  const [currentTab, setCurrentTab] = useState(0);
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchSystemStatus = async () => {
    try {
      setRefreshing(true);
      const status = await apiService.getSystemStatus();
      
      // Get model information
      try {
        const modelInfo = await apiService.getModelInfo();
        status.model_info = modelInfo.data;
      } catch (modelErr) {
        console.warn('Failed to load model info:', modelErr);
      }
      
      setSystemStatus(status);
      setError(null);
    } catch (err) {
      setError('Failed to connect to the backend. Please ensure the API server is running.');
      console.error('Error fetching system status:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchSystemStatus();
    // Set up periodic status checks
    const interval = setInterval(fetchSystemStatus, 30000); // Every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

  const handleRefresh = () => {
    fetchSystemStatus();
  };

  if (loading) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Container maxWidth="sm" sx={{ mt: 8, textAlign: 'center' }}>
          <CircularProgress size={60} />
          <Typography variant="h6" sx={{ mt: 2 }}>
            Connecting to Reddit Knowledge Base...
          </Typography>
        </Container>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static" elevation={2}>
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Reddit Knowledge Base
            </Typography>
            <Typography variant="body2" sx={{ mr: 2 }}>
              {systemStatus ? 
                `Ready • ${systemStatus.chatbot?.knowledge_base?.total_documents || 0} docs` :
                'Connecting...'
              }
            </Typography>
            <Typography variant="caption" sx={{ mr: 2, opacity: 0.7 }}>
              {systemStatus?.model_info?.current_model || 'Model: Unknown'}
            </Typography>
          </Toolbar>
        </AppBar>

        {error && (
          <Container maxWidth="lg" sx={{ mt: 2 }}>
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          </Container>
        )}

        <Container maxWidth="lg" sx={{ mt: 3 }}>
          <Paper elevation={3}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs 
                value={currentTab} 
                onChange={handleTabChange} 
                aria-label="navigation tabs"
                centered
              >
                <Tab 
                  icon={<ChatIcon />} 
                  label="Chat" 
                  id="tab-0" 
                  aria-controls="tabpanel-0" 
                />
                <Tab 
                  icon={<InsightsIcon />} 
                  label="Insights" 
                  id="tab-1" 
                  aria-controls="tabpanel-1" 
                />
                <Tab 
                  icon={<SettingsIcon />} 
                  label="Configuration" 
                  id="tab-2" 
                  aria-controls="tabpanel-2" 
                />
              </Tabs>
            </Box>

            <TabPanel value={currentTab} index={0}>
              <ChatInterface systemStatus={systemStatus} />
            </TabPanel>

            <TabPanel value={currentTab} index={1}>
              <InsightsDashboard systemStatus={systemStatus} />
            </TabPanel>

            <TabPanel value={currentTab} index={2}>
              <ConfigurationPanel 
                systemStatus={systemStatus} 
                onStatusUpdate={fetchSystemStatus}
              />
            </TabPanel>
          </Paper>
        </Container>

        {/* Refresh FAB */}
        <Fab
          color="primary"
          aria-label="refresh"
          sx={{
            position: 'fixed',
            bottom: 16,
            right: 16,
          }}
          onClick={handleRefresh}
          disabled={refreshing}
        >
          {refreshing ? <CircularProgress size={24} /> : <RefreshIcon />}
        </Fab>
      </Box>
    </ThemeProvider>
  );
}

export default App;
