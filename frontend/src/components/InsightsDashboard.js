import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Button,
  CircularProgress,
  Alert,
  Paper,
  List,
  ListItem,
  ListItemText,
  Divider,
  LinearProgress,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  Psychology as PsychologyIcon,
  Timeline as TimelineIcon,
  CloudDownload as CloudDownloadIcon,
  Visibility as VisibilityIcon
} from '@mui/icons-material';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as ChartTooltip, ResponsiveContainer } from 'recharts';
import { insightUtils, workflowUtils } from '../services/apiService';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

const InsightsDashboard = ({ systemStatus }) => {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedInsight, setSelectedInsight] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isCollecting, setIsCollecting] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const data = await insightUtils.getDashboardData();
      setDashboardData(data);
      setError(null);
    } catch (err) {
      console.error('Error loading dashboard data:', err);
      setError('Failed to load insights dashboard');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateInsights = async () => {
    try {
      setIsGenerating(true);
      const result = await insightUtils.generate();
      if (result.success) {
        // Refresh dashboard after generation
        setTimeout(() => {
          loadDashboardData();
        }, 2000);
      } else {
        setError(result.error);
      }
    } catch (err) {
      setError('Failed to generate insights');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRunWorkflow = async () => {
    try {
      setIsCollecting(true);
      const result = await workflowUtils.runFullWorkflow();
      if (result.success) {
        // Refresh dashboard after workflow
        setTimeout(() => {
          loadDashboardData();
        }, 5000);
      } else {
        setError(result.error);
      }
    } catch (err) {
      setError('Failed to run workflow');
    } finally {
      setIsCollecting(false);
    }
  };

  const handleViewDetails = (insight) => {
    setSelectedInsight(insight);
    setDetailsOpen(true);
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString([], {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const prepareSentimentData = (sentiment) => {
    return Object.entries(sentiment || {}).map(([key, value]) => ({
      name: key.charAt(0).toUpperCase() + key.slice(1),
      value: Math.round(value * 100),
      count: value
    }));
  };

  const prepareTopicData = (clusters) => {
    return (clusters || []).slice(0, 10).map(cluster => ({
      name: cluster.name || 'Unknown',
      documents: cluster.document_count || 0,
      score: Math.round(cluster.avg_score || 0)
    }));
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading insights...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
        <Button onClick={loadDashboardData} sx={{ ml: 2 }}>
          Retry
        </Button>
      </Alert>
    );
  }

  const recentInsights = dashboardData?.recent_insights || [];
  const knowledgeBaseStats = dashboardData?.knowledge_base_stats || {};
  const latestInsight = recentInsights[0];

  return (
    <Box>
      {/* Header Actions */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Insights Dashboard
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={loadDashboardData}
            disabled={loading}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<CloudDownloadIcon />}
            onClick={handleRunWorkflow}
            disabled={isCollecting}
            color="primary"
          >
            {isCollecting ? <CircularProgress size={20} /> : 'Run Full Workflow'}
          </Button>
          <Button
            variant="contained"
            startIcon={<PsychologyIcon />}
            onClick={handleGenerateInsights}
            disabled={isGenerating}
            color="secondary"
          >
            {isGenerating ? <CircularProgress size={20} /> : 'Generate Insights'}
          </Button>
        </Box>
      </Box>

      {/* Status Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Documents
              </Typography>
              <Typography variant="h4">
                {knowledgeBaseStats.total_documents || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Available Insights
              </Typography>
              <Typography variant="h4">
                {recentInsights.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Subreddits Covered
              </Typography>
              <Typography variant="h4">
                {dashboardData?.available_subreddits?.length || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Last Updated
              </Typography>
              <Typography variant="body2">
                {dashboardData?.last_updated ? 
                  formatDate(dashboardData.last_updated) : 
                  'Never'
                }
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {recentInsights.length === 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <PsychologyIcon sx={{ fontSize: 64, color: 'grey.400', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            No insights available
          </Typography>
          <Typography color="textSecondary" sx={{ mb: 3 }}>
            Run the data collection workflow to generate insights from Reddit discussions.
          </Typography>
          <Button
            variant="contained"
            onClick={handleRunWorkflow}
            disabled={isCollecting}
            size="large"
          >
            {isCollecting ? <CircularProgress size={20} /> : 'Start Data Collection'}
          </Button>
        </Paper>
      ) : (
        <Grid container spacing={3}>
          {/* Latest Insights Overview */}
          <Grid item xs={12} lg={8}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">
                    Latest Analysis Results
                  </Typography>
                  {latestInsight && (
                    <Chip 
                      label={formatDate(latestInsight.created_at)} 
                      size="small" 
                      color="primary" 
                    />
                  )}
                </Box>

                {latestInsight && (
                  <>
                    {/* Key Insights */}
                    <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                      Key Insights:
                    </Typography>
                    <List dense>
                      {(latestInsight.key_insights || []).slice(0, 5).map((insight, index) => (
                        <ListItem key={index}>
                          <ListItemText primary={`• ${insight}`} />
                        </ListItem>
                      ))}
                    </List>

                    {/* Top Topics */}
                    <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                      Trending Topics:
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                      {(latestInsight.trending_topics || []).slice(0, 8).map((topic, index) => (
                        <Chip 
                          key={index} 
                          label={topic} 
                          size="small" 
                          variant="outlined"
                          icon={<TrendingUpIcon />}
                        />
                      ))}
                    </Box>

                    <Button
                      variant="outlined"
                      startIcon={<VisibilityIcon />}
                      onClick={() => handleViewDetails(latestInsight)}
                      sx={{ mt: 1 }}
                    >
                      View Full Details
                    </Button>
                  </>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Sentiment Analysis */}
          <Grid item xs={12} lg={4}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Overall Sentiment
                </Typography>
                {latestInsight?.overall_sentiment && (
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie
                        data={prepareSentimentData(latestInsight.overall_sentiment)}
                        cx="50%"
                        cy="50%"
                        innerRadius={40}
                        outerRadius={80}
                        dataKey="value"
                      >
                        {prepareSentimentData(latestInsight.overall_sentiment).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <ChartTooltip formatter={(value) => `${value}%`} />
                    </PieChart>
                  </ResponsiveContainer>
                )}
                {latestInsight?.overall_sentiment && (
                  <Box sx={{ mt: 1 }}>
                    {prepareSentimentData(latestInsight.overall_sentiment).map((item, index) => (
                      <Box key={item.name} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                        <Box 
                          sx={{ 
                            width: 12, 
                            height: 12, 
                            backgroundColor: COLORS[index % COLORS.length],
                            mr: 1,
                            borderRadius: '50%'
                          }} 
                        />
                        <Typography variant="body2">
                          {item.name}: {item.value}%
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Topic Distribution */}
          {latestInsight?.clusters && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Topic Distribution
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={prepareTopicData(latestInsight.clusters)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="name" 
                        angle={-45}
                        textAnchor="end"
                        height={100}
                      />
                      <YAxis />
                      <ChartTooltip />
                      <Bar dataKey="documents" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Recent Insights History */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Insights History
                </Typography>
                <List>
                  {recentInsights.slice(0, 5).map((insight, index) => (
                    <React.Fragment key={insight.id}>
                      <ListItem
                        secondaryAction={
                          <Button
                            size="small"
                            onClick={() => handleViewDetails(insight)}
                          >
                            View Details
                          </Button>
                        }
                      >
                        <ListItemText
                          primary={`Analysis ${insight.id}`}
                          secondary={
                            <Box>
                              <Typography variant="body2" color="textSecondary">
                                {formatDate(insight.created_at)} • 
                                {insight.total_documents} documents • 
                                {insight.clusters?.length || 0} topics
                              </Typography>
                              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
                                {(insight.subreddits || []).map((subreddit) => (
                                  <Chip 
                                    key={subreddit} 
                                    label={`r/${subreddit}`} 
                                    size="small" 
                                    variant="outlined"
                                  />
                                ))}
                              </Box>
                            </Box>
                          }
                        />
                      </ListItem>
                      {index < recentInsights.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Details Dialog */}
      <Dialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Insight Details - {selectedInsight?.id}
        </DialogTitle>
        <DialogContent>
          {selectedInsight && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Analysis Date: {formatDate(selectedInsight.created_at)}
              </Typography>
              <Typography variant="subtitle2" gutterBottom>
                Documents Analyzed: {selectedInsight.total_documents}
              </Typography>
              <Typography variant="subtitle2" gutterBottom>
                Topics Found: {selectedInsight.clusters?.length || 0}
              </Typography>

              <Divider sx={{ my: 2 }} />

              <Typography variant="h6" gutterBottom>
                Key Insights
              </Typography>
              <List>
                {(selectedInsight.key_insights || []).map((insight, index) => (
                  <ListItem key={index}>
                    <ListItemText primary={`• ${insight}`} />
                  </ListItem>
                ))}
              </List>

              <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                Top Topics
              </Typography>
              {(selectedInsight.clusters || []).slice(0, 5).map((cluster, index) => (
                <Card key={index} variant="outlined" sx={{ mb: 2 }}>
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      {cluster.name} ({cluster.document_count} discussions)
                    </Typography>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                      {cluster.description}
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
                      {(cluster.keywords || []).slice(0, 8).map((keyword, kidx) => (
                        <Chip key={kidx} label={keyword} size="small" variant="outlined" />
                      ))}
                    </Box>
                  </CardContent>
                </Card>
              ))}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default InsightsDashboard;
