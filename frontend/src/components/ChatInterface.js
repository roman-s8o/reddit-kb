import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  Avatar,
  Chip,
  CircularProgress,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Tooltip,
  Card,
  CardContent,
  Divider,
  Link
} from '@mui/material';
import {
  Send as SendIcon,
  Person as PersonIcon,
  SmartToy as BotIcon,
  Clear as ClearIcon,
  ContentCopy as CopyIcon,
  ThumbUp as ThumbUpIcon,
  Reddit as RedditIcon
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import { chatUtils, apiService } from '../services/apiService';

const ChatInterface = ({ systemStatus }) => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedSubreddits, setSelectedSubreddits] = useState([]);
  const [availableSubreddits, setAvailableSubreddits] = useState([]);
  const [suggestions, setSuggestions] = useState([]);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  // Scroll to bottom when messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load initial data
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      // Get configuration for available subreddits
      const config = await apiService.getConfiguration();
      setAvailableSubreddits(config.subreddits || []);

      // Get initial suggestions
      const suggestions = await chatUtils.getSuggestions();
      setSuggestions(suggestions);

      // Add welcome message
      setMessages([{
        id: Date.now(),
        type: 'bot',
        content: 'Hello! I\'m your Reddit Knowledge Base assistant. I can help you explore discussions, find insights, and answer questions based on Reddit data. What would you like to know?',
        timestamp: new Date().toISOString(),
        sources: []
      }]);
    } catch (err) {
      console.error('Error loading initial data:', err);
      setError('Failed to load initial data');
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage.trim(),
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setError(null);

    try {
      const result = await chatUtils.askQuestion(inputMessage.trim(), {
        subreddits: selectedSubreddits.length > 0 ? selectedSubreddits : null,
        includeInsights: true,
        maxResults: 10
      });

      if (result.success) {
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: result.data.answer,
          timestamp: result.data.timestamp,
          sources: result.data.references || [],
          insights: result.data.insights || [],
          confidence: result.data.confidence
        };

        setMessages(prev => [...prev, botMessage]);

        // Get new suggestions based on current subreddit selection
        const newSuggestions = await chatUtils.getSuggestions(selectedSubreddits.length > 0 ? selectedSubreddits : null);
        setSuggestions(newSuggestions);
      } else {
        throw new Error(result.error);
      }
    } catch (err) {
      console.error('Error sending message:', err);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'I apologize, but I encountered an error while processing your question. Please try again.',
        timestamp: new Date().toISOString(),
        sources: [],
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setInputMessage(suggestion);
  };

  const handleClearChat = () => {
    setMessages([{
      id: Date.now(),
      type: 'bot',
      content: 'Chat cleared. How can I help you today?',
      timestamp: new Date().toISOString(),
      sources: []
    }]);
  };

  const handleCopyMessage = (content) => {
    navigator.clipboard.writeText(content);
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const renderMessage = (message) => {
    const isUser = message.type === 'user';
    
    return (
      <Box
        key={message.id}
        sx={{
          display: 'flex',
          flexDirection: isUser ? 'row-reverse' : 'row',
          mb: 2,
          alignItems: 'flex-start'
        }}
      >
        <Avatar
          sx={{
            bgcolor: isUser ? 'primary.main' : 'secondary.main',
            mx: 1
          }}
        >
          {isUser ? <PersonIcon /> : <BotIcon />}
        </Avatar>
        
        <Paper
          elevation={2}
          sx={{
            p: 2,
            maxWidth: '70%',
            bgcolor: isUser ? 'primary.light' : 'grey.100',
            color: isUser ? 'primary.contrastText' : 'text.primary'
          }}
        >
          <Box sx={{ mb: 1 }}>
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </Box>
          
          {message.confidence && (
            <Chip
              size="small"
              label={`Confidence: ${(message.confidence * 100).toFixed(0)}%`}
              color={message.confidence > 0.7 ? 'success' : 'warning'}
              sx={{ mb: 1 }}
            />
          )}

          {message.sources && message.sources.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="caption" color="text.secondary">
                Sources:
              </Typography>
              {message.sources.slice(0, 3).map((source, index) => (
                <Card key={index} variant="outlined" sx={{ mt: 1, p: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <RedditIcon sx={{ mr: 1, fontSize: 16 }} />
                    <Typography variant="caption">
                      r/{source.subreddit} • 
                      {source.source_type === 'post' ? 'Post' : 'Comment'} • 
                      Similarity: {(source.similarity_score * 100).toFixed(0)}%
                    </Typography>
                  </Box>
                  {source.title && (
                    <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 0.5 }}>
                      {source.title}
                    </Typography>
                  )}
                  {source.url && (
                    <Link href={source.url} target="_blank" rel="noopener" sx={{ fontSize: '0.75rem', display: 'block', mb: 0.5 }}>
                      View on Reddit
                    </Link>
                  )}
                </Card>
              ))}
            </Box>
          )}

          {message.insights && message.insights.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="caption" color="text.secondary">
                Related Insights:
              </Typography>
              {message.insights.slice(0, 2).map((insight, index) => (
                <Card key={index} variant="outlined" sx={{ mt: 1, p: 1, bgcolor: 'info.light', color: 'info.contrastText' }}>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block' }}>
                    {insight.topic} ({insight.document_count} docs)
                  </Typography>
                  <Typography variant="caption">
                    Keywords: {insight.keywords.join(', ')}
                  </Typography>
                  <Typography variant="caption" sx={{ display: 'block', mt: 0.5 }}>
                    Relevance: {(insight.relevance * 100).toFixed(0)}%
                  </Typography>
                </Card>
              ))}
            </Box>
          )}

          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            mt: 1
          }}>
            <Typography variant="caption" color="text.secondary">
              {formatTimestamp(message.timestamp)}
            </Typography>
            
            <Box>
              <Tooltip title="Copy message">
                <IconButton
                  size="small"
                  onClick={() => handleCopyMessage(message.content)}
                >
                  <CopyIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </Paper>
      </Box>
    );
  };

  return (
    <Box sx={{ height: '70vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header with controls */}
      <Box sx={{ mb: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
        <FormControl size="small" sx={{ minWidth: 200 }}>
          <InputLabel>Filter Subreddits</InputLabel>
          <Select
            multiple
            value={selectedSubreddits}
            onChange={(e) => setSelectedSubreddits(e.target.value)}
            label="Filter Subreddits"
            renderValue={(selected) => (
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {selected.map((value) => (
                  <Chip key={value} label={`r/${value}`} size="small" />
                ))}
              </Box>
            )}
          >
            {availableSubreddits.map((subreddit) => (
              <MenuItem key={subreddit} value={subreddit}>
                r/{subreddit}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <Button
          variant="outlined"
          startIcon={<ClearIcon />}
          onClick={handleClearChat}
          size="small"
        >
          Clear Chat
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Messages area */}
      <Paper
        variant="outlined"
        sx={{
          flex: 1,
          p: 2,
          overflowY: 'auto',
          backgroundColor: 'background.paper'
        }}
      >
        {messages.map(renderMessage)}
        
        {isLoading && (
          <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
            <Avatar sx={{ bgcolor: 'secondary.main', mx: 1 }}>
              <BotIcon />
            </Avatar>
            <Paper elevation={2} sx={{ p: 2, display: 'flex', alignItems: 'center' }}>
              <CircularProgress size={20} sx={{ mr: 2 }} />
              <Typography>Thinking...</Typography>
            </Paper>
          </Box>
        )}
        
        <div ref={messagesEndRef} />
      </Paper>

      {/* Suggestions */}
      {suggestions.length > 0 && (
        <Box sx={{ mt: 2, mb: 1 }}>
          <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
            Suggested questions:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {suggestions.slice(0, 5).map((suggestion, index) => (
              <Chip
                key={index}
                label={suggestion}
                onClick={() => handleSuggestionClick(suggestion)}
                size="small"
                variant="outlined"
                clickable
              />
            ))}
          </Box>
        </Box>
      )}

      {/* Input area */}
      <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
        <TextField
          fullWidth
          multiline
          maxRows={4}
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask me anything about Reddit discussions..."
          disabled={isLoading}
          variant="outlined"
        />
        <Button
          variant="contained"
          endIcon={<SendIcon />}
          onClick={handleSendMessage}
          disabled={!inputMessage.trim() || isLoading}
          sx={{ px: 3 }}
        >
          Send
        </Button>
      </Box>

      {/* Status info */}
      {systemStatus && (
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
          Knowledge base: {systemStatus.chatbot?.knowledge_base?.total_documents || 0} documents
          {selectedSubreddits.length > 0 && 
            ` • Filtering: ${selectedSubreddits.map(s => `r/${s}`).join(', ')}`
          }
        </Typography>
      )}
    </Box>
  );
};

export default ChatInterface;
