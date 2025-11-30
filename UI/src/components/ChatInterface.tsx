import { useState, useRef, useEffect, useContext } from 'react';
import { 
  Box, 
  TextField, 
  IconButton, 
  Paper, 
  Typography, 
  Avatar, 
  List, 
  ListItem, 
  ListItemAvatar,
  AppBar,
  Toolbar,
  Container,
  useTheme,
  Tooltip,
  Fade
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { ThemeContext } from '../App';

type Message = {
  id: number;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  isLoading?: boolean;
};

const ChatInterface: React.FC = () => {
  // Theme context is available if needed
  useContext(ThemeContext);
  const theme = useTheme();
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: 'Hello! I\'m your AI assistant. How can I help you today?',
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-focus input on mount
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  const simulateTyping = (callback: () => void) => {
    setIsTyping(true);
    setTimeout(() => {
      callback();
      setIsTyping(false);
    }, 1000);
  };

  const handleSend = () => {
    const messageText = input.trim();
    if (messageText === '') return;

    // Add user message
    const userMessage: Message = {
      id: Date.now(),
      text: messageText,
      sender: 'user',
      timestamp: new Date()
    };

    // Add loading state for bot
    const loadingMessage: Message = {
      id: Date.now() + 1,
      text: '',
      sender: 'bot',
      timestamp: new Date(),
      isLoading: true
    };

    const newMessages = [...messages, userMessage, loadingMessage];
    setMessages(newMessages);
    setInput('');

    // Simulate bot response with typing effect
    simulateTyping(() => {
      const botMessage: Message = {
        id: Date.now() + 2,
        text: `I received your message: "${messageText}"`,
        sender: 'bot',
        timestamp: new Date()
      };
      
      // Replace loading message with actual response
      setMessages(prev => {
        const updated = [...prev];
        const loadingIndex = updated.findIndex(m => m.isLoading);
        if (loadingIndex !== -1) {
          updated[loadingIndex] = botMessage;
        }
        return updated;
      });
    });
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Format time for message timestamps
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100vh',
      backgroundColor: 'background.default',
      transition: theme.transitions.create('background-color'),
      position: 'relative',
      overflow: 'hidden'
    }}>
      <AppBar 
        position="static" 
        color="transparent" 
        elevation={0}
        sx={{
          background: 'transparent',
          backdropFilter: 'blur(10px)',
          borderBottom: `1px solid ${theme.palette.divider}`,
          zIndex: 1
        }}
      >
        <Toolbar sx={{ px: { xs: 2, sm: 4 } }}>
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center',
            background: theme.palette.mode === 'dark' 
              ? 'linear-gradient(90deg, rgba(37,99,235,0.2) 0%, rgba(79,70,229,0.2) 100%)' 
              : 'linear-gradient(90deg, rgba(37,99,235,0.1) 0%, rgba(79,70,229,0.1) 100%)',
            borderRadius: 3,
            px: 2,
            py: 1
          }}>
            <AutoAwesomeIcon 
              sx={{ 
                color: theme.palette.primary.main,
                mr: 1,
                fontSize: 20
              }} 
            />
            <Typography 
              variant="h6" 
              sx={{ 
                fontWeight: 600,
                background: theme.palette.mode === 'dark'
                  ? 'linear-gradient(90deg, #3b82f6 0%, #818cf8 100%)'
                  : 'linear-gradient(90deg, #2563eb 0%, #4f46e5 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                textFillColor: 'transparent',
                fontSize: { xs: '1rem', sm: '1.25rem' }
              }}
            >
              EchoChat AI
            </Typography>
          </Box>
        </Toolbar>
      </AppBar>

      <Container 
        maxWidth="md" 
        sx={{ 
          flexGrow: 1, 
          overflow: 'auto', 
          py: 2,
          px: { xs: 1.5, sm: 2 },
          '&::-webkit-scrollbar': {
            width: '6px',
          },
          '&::-webkit-scrollbar-track': {
            background: 'transparent',
          },
          '&::-webkit-scrollbar-thumb': {
            background: theme.palette.mode === 'dark' ? '#4b5563' : '#d1d5db',
            borderRadius: '3px',
          },
          '&::-webkit-scrollbar-thumb:hover': {
            background: theme.palette.mode === 'dark' ? '#6b7280' : '#9ca3af',
          },
        }}
      >
        <List sx={{ width: '100%', pb: 2 }}>
          {messages.map((message) => (
            <Fade in={true} key={message.id} timeout={300}>
              <ListItem 
                alignItems="flex-start"
                sx={{
                  flexDirection: message.sender === 'user' ? 'row-reverse' : 'row',
                  mb: 2,
                  px: { xs: 0.5, sm: 2 },
                  '&:hover': {
                    '& .message-timestamp': {
                      opacity: 1,
                    },
                  },
                }}
              >
                <ListItemAvatar 
                  sx={{
                    minWidth: 40,
                    alignSelf: 'flex-end',
                    mb: 1,
                    ...(message.sender === 'user' ? { ml: 1 } : { mr: 1 })
                  }}
                >
                  <Tooltip 
                    title={message.sender === 'user' ? 'You' : 'AI Assistant'}
                    placement={message.sender === 'user' ? 'left' : 'right'}
                    arrow
                  >
                    <Avatar 
                      sx={{ 
                        bgcolor: message.sender === 'user' 
                          ? theme.palette.primary.main 
                          : theme.palette.secondary.main,
                        width: 36,
                        height: 36,
                        '&:hover': {
                          transform: 'scale(1.05)',
                          transition: 'transform 0.2s',
                        },
                      }}
                    >
                      {message.sender === 'user' ? (
                        <PersonIcon fontSize="small" />
                      ) : (
                        <SmartToyIcon fontSize="small" />
                      )}
                    </Avatar>
                  </Tooltip>
                </ListItemAvatar>
                
                <Box 
                  sx={{
                    maxWidth: { xs: 'calc(100% - 60px)', sm: '70%' },
                    position: 'relative',
                    ...(message.sender === 'user' && {
                      ml: { xs: 0, sm: 2 },
                      mr: { xs: 0.5, sm: 0 },
                    }),
                  }}
                >
                  <Paper 
                    elevation={0}
                    sx={{
                      p: 2,
                      backgroundColor: message.sender === 'user' 
                        ? theme.palette.mode === 'dark'
                          ? theme.palette.primary.dark
                          : theme.palette.primary.main
                        : theme.palette.background.paper,
                      color: message.sender === 'user' 
                        ? theme.palette.primary.contrastText 
                        : theme.palette.text.primary,
                      borderRadius: message.sender === 'user' 
                        ? '18px 18px 0 18px' 
                        : '18px 18px 18px 0',
                      border: `1px solid ${theme.palette.divider}`,
                      position: 'relative',
                      overflow: 'hidden',
                      '&:before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        height: '1px',
                        background: `linear-gradient(90deg, transparent, ${theme.palette.divider}, transparent)`,
                        opacity: 0.5,
                      },
                    }}
                  >
                    {message.isLoading ? (
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        {[...Array(3)].map((_, i) => (
                          <Box
                            key={i}
                            sx={{
                              width: 8,
                              height: 8,
                              borderRadius: '50%',
                              backgroundColor: theme.palette.text.secondary,
                              opacity: 0.7,
                              animation: 'pulse 1.4s infinite',
                              animationDelay: `${i * 0.2}s`,
                              '@keyframes pulse': {
                                '0%, 100%': { transform: 'translateY(0)', opacity: 0.4 },
                                '50%': { transform: 'translateY(-5px)', opacity: 1 },
                              },
                            }}
                          />
                        ))}
                      </Box>
                    ) : (
                      <>
                        <Typography 
                          variant="body1" 
                          sx={{
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word',
                            lineHeight: 1.6,
                            '& a': {
                              color: message.sender === 'user' 
                                ? theme.palette.primary.light 
                                : theme.palette.primary.main,
                              textDecoration: 'none',
                              '&:hover': {
                                textDecoration: 'underline',
                              },
                            },
                            '& code': {
                              fontFamily: 'monospace',
                              backgroundColor: message.sender === 'user'
                                ? 'rgba(255, 255, 255, 0.1)'
                                : 'rgba(0, 0, 0, 0.05)',
                              padding: '0.2em 0.4em',
                              borderRadius: 4,
                              fontSize: '0.9em',
                            },
                          }}
                          dangerouslySetInnerHTML={{ 
                            __html: message.text.replace(/\n/g, '<br />') 
                          }}
                        />
                        <Typography 
                          variant="caption" 
                          className="message-timestamp"
                          sx={{
                            display: 'block',
                            textAlign: message.sender === 'user' ? 'right' : 'left',
                            mt: 1,
                            opacity: 0,
                            transition: 'opacity 0.2s',
                            fontSize: '0.7rem',
                            color: message.sender === 'user' 
                              ? 'rgba(255, 255, 255, 0.6)' 
                              : theme.palette.text.secondary,
                          }}
                        >
                          {formatTime(message.timestamp)}
                        </Typography>
                      </>
                    )}
                  </Paper>
                </Box>
              </ListItem>
            </Fade>
          ))}
          <div ref={messagesEndRef} />
        </List>
      </Container>

      <Box 
        component="footer" 
        sx={{ 
          p: 2,
          pt: 1.5,
          borderTop: `1px solid ${theme.palette.divider}`,
          backgroundColor: 'background.paper',
          position: 'relative',
          '&:before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '30px',
            background: `linear-gradient(to bottom, transparent, ${theme.palette.background.default} 80%)`,
            pointerEvents: 'none',
            transform: 'translateY(-100%)',
          },
        }}
      >
        <Box 
          sx={{ 
            display: 'flex', 
            alignItems: 'flex-end', 
            maxWidth: 'md', 
            mx: 'auto',
            position: 'relative',
          }}
        >
          <TextField
            inputRef={inputRef}
            fullWidth
            variant="outlined"
            placeholder="Message EchoChat..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            multiline
            maxRows={6}
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: '24px',
                backgroundColor: 'background.paper',
                border: `1px solid ${theme.palette.divider}`,
                transition: theme.transitions.create(['border-color', 'box-shadow']),
                '&:hover .MuiOutlinedInput-notchedOutline': {
                  borderColor: theme.palette.primary.main,
                },
                '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                  borderColor: theme.palette.primary.main,
                  borderWidth: '1px',
                },
                '& textarea': {
                  maxHeight: '200px',
                  overflowY: 'auto !important',
                  paddingRight: '50px',
                  '&::-webkit-scrollbar': {
                    width: '6px',
                  },
                  '&::-webkit-scrollbar-track': {
                    background: 'transparent',
                  },
                  '&::-webkit-scrollbar-thumb': {
                    background: theme.palette.mode === 'dark' ? '#4b5563' : '#d1d5db',
                    borderRadius: '3px',
                  },
                },
              },
              '& .MuiOutlinedInput-input': {
                padding: '12px 20px',
                '&::placeholder': {
                  color: theme.palette.text.secondary,
                  opacity: 0.8,
                },
              },
              '& .MuiOutlinedInput-notchedOutline': {
                border: 'none',
              },
              '&:hover': {
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: theme.palette.divider,
                },
              },
            }}
          />
          <IconButton 
            color="primary" 
            onClick={handleSend}
            disabled={!input.trim() || isTyping}
            sx={{ 
              position: 'absolute',
              right: '8px',
              bottom: '8px',
              width: '40px',
              height: '40px',
              backgroundColor: input.trim() ? 'primary.main' : 'transparent',
              color: input.trim() ? 'white' : theme.palette.text.secondary,
              transition: theme.transitions.create(['background-color', 'transform', 'box-shadow']),
              '&:hover': {
                backgroundColor: input.trim() ? 'primary.dark' : 'action.hover',
                transform: input.trim() ? 'translateY(-2px)' : 'none',
                boxShadow: input.trim() ? '0 4px 12px rgba(37, 99, 235, 0.2)' : 'none',
              },
              '&:active': {
                transform: 'scale(0.95)',
              },
              '&:disabled': {
                backgroundColor: 'transparent',
                color: theme.palette.action.disabled,
              },
            }}
          >
            {isTyping ? (
              <Box 
                sx={{
                  width: 24,
                  height: 24,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Box 
                  sx={{
                    width: 16,
                    height: 16,
                    border: `2px solid ${theme.palette.primary.main}`,
                    borderTopColor: 'transparent',
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite',
                    '@keyframes spin': {
                      '0%': { transform: 'rotate(0deg)' },
                      '100%': { transform: 'rotate(360deg)' },
                    },
                  }}
                />
              </Box>
            ) : (
              <SendIcon fontSize="small" />
            )}
          </IconButton>
        </Box>
        <Typography 
          variant="caption" 
          sx={{
            display: 'block',
            textAlign: 'center',
            mt: 1,
            color: 'text.secondary',
            fontSize: '0.7rem',
            opacity: 0.7,
          }}
        >
          EchoChat may produce inaccurate information. Consider verifying important details.
        </Typography>
      </Box>
    </Box>
  );
};

export default ChatInterface;
