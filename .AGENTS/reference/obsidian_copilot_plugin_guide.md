# Building Obsidian Co-Pilot Plugin with Custom AI Model

**Purpose**: Use this guide when building an Obsidian plugin that integrates with Co-Pilot functionality using your own AI model via OpenAI-compatible endpoint.

## Overall Pattern

```
┌─────────────────────────────────────────────────┐
│           Obsidian Plugin Architecture          │
├─────────────────────────────────────────────────┤
│  Plugin Class (extends Plugin)                  │
│  ├─ Settings Interface & Tab                    │
│  ├─ AI Model Configuration                      │
│  ├─ Commands & Events                           │
│  └─ Agent/Chat Interface                        │
└─────────────────────────────────────────────────┘
         │
         ├─► OpenAI-Compatible API Endpoint
         ├─► Obsidian Vault (read/write notes)
         └─► UI Components (Modal, StatusBar, etc.)
```

**Key Architecture**: Plugin extends Obsidian's `Plugin` class, implements settings for custom endpoint configuration, registers commands for user interaction, and maintains stateful connection to AI model through OpenAI-compatible API.

## Step 1: Plugin Setup & Structure

Create the main plugin class with TypeScript:

```typescript
import { Plugin, PluginSettingTab, Setting, Notice } from 'obsidian';

interface CopilotSettings {
  apiEndpoint: string;
  apiKey: string;
  modelName: string;
  maxTokens: number;
  temperature: number;
}

const DEFAULT_SETTINGS: CopilotSettings = {
  apiEndpoint: 'http://localhost:11434/v1',
  apiKey: '',
  modelName: 'gpt-3.5-turbo',
  maxTokens: 2048,
  temperature: 0.7
}

export default class CopilotPlugin extends Plugin {
  settings: CopilotSettings;

  async onload() {
    await this.loadSettings();

    // Register commands
    this.addCommand({
      id: 'open-copilot-chat',
      name: 'Open Copilot Chat',
      callback: () => this.openChatModal()
    });

    // Add settings tab
    this.addSettingTab(new CopilotSettingTab(this.app, this));
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }

  async saveSettings() {
    await this.saveData(this.settings);
  }
}
```

**Critical rules**:
- Extend `Plugin` base class from Obsidian API
- Use `async onload()` for initialization
- Store settings with `loadData()`/`saveData()` for persistence
- Register all commands in `onload()` method
- Always provide default settings

## Step 2: Settings UI Configuration

Implement settings tab for custom endpoint configuration:

```typescript
class CopilotSettingTab extends PluginSettingTab {
  plugin: CopilotPlugin;

  constructor(app: App, plugin: CopilotPlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();

    new Setting(containerEl)
      .setName('API Endpoint')
      .setDesc('OpenAI-compatible API endpoint (e.g., http://localhost:11434/v1)')
      .addText(text => text
        .setPlaceholder('http://localhost:11434/v1')
        .setValue(this.plugin.settings.apiEndpoint)
        .onChange(async (value) => {
          this.plugin.settings.apiEndpoint = value;
          await this.plugin.saveSettings();
        }));

    new Setting(containerEl)
      .setName('API Key')
      .setDesc('API key for authentication (leave empty for local models)')
      .addText(text => {
        text.setPlaceholder('sk-...')
          .setValue(this.plugin.settings.apiKey)
          .onChange(async (value) => {
            this.plugin.settings.apiKey = value;
            await this.plugin.saveSettings();
          });
        text.inputEl.type = 'password';
      });

    new Setting(containerEl)
      .setName('Model Name')
      .setDesc('Model identifier (e.g., gpt-3.5-turbo, llama2)')
      .addText(text => text
        .setPlaceholder('gpt-3.5-turbo')
        .setValue(this.plugin.settings.modelName)
        .onChange(async (value) => {
          this.plugin.settings.modelName = value;
          await this.plugin.saveSettings();
        }));

    new Setting(containerEl)
      .setName('Max Tokens')
      .setDesc('Maximum tokens for model response')
      .addText(text => text
        .setPlaceholder('2048')
        .setValue(String(this.plugin.settings.maxTokens))
        .onChange(async (value) => {
          this.plugin.settings.maxTokens = parseInt(value) || 2048;
          await this.plugin.saveSettings();
        }));
  }
}
```

**Critical rules**:
- Extend `PluginSettingTab` for settings UI
- Use `new Setting()` for each configuration field
- Save settings on every change with `await saveSettings()`
- Use `text.inputEl.type = 'password'` for sensitive fields
- Provide clear descriptions for each setting

## Step 3: OpenAI-Compatible API Integration

Create API client for OpenAI-compatible endpoints:

```typescript
interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
}

class AIClient {
  private settings: CopilotSettings;

  constructor(settings: CopilotSettings) {
    this.settings = settings;
  }

  async chat(messages: ChatMessage[]): Promise<string> {
    const requestBody: ChatCompletionRequest = {
      model: this.settings.modelName,
      messages: messages,
      max_tokens: this.settings.maxTokens,
      temperature: this.settings.temperature,
      stream: false
    };

    try {
      const response = await fetch(
        `${this.settings.apiEndpoint}/chat/completions`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(this.settings.apiKey && {
              'Authorization': `Bearer ${this.settings.apiKey}`
            })
          },
          body: JSON.stringify(requestBody)
        }
      );

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      return data.choices[0].message.content;
    } catch (error) {
      new Notice(`Failed to connect to AI: ${error.message}`);
      throw error;
    }
  }
}
```

**Critical rules**:
- Use `fetch()` for HTTP requests to API endpoint
- Follow OpenAI `/v1/chat/completions` format
- Include `Authorization` header only if API key exists
- Handle errors with try/catch and show user notices
- Return plain string response from API

## Step 4: Chat Interface Modal

Build interactive chat modal for user interaction:

```typescript
import { Modal, TextAreaComponent } from 'obsidian';

class CopilotChatModal extends Modal {
  private plugin: CopilotPlugin;
  private client: AIClient;
  private chatHistory: ChatMessage[] = [];
  private chatContainer: HTMLElement;
  private inputArea: TextAreaComponent;

  constructor(app: App, plugin: CopilotPlugin) {
    super(app);
    this.plugin = plugin;
    this.client = new AIClient(plugin.settings);
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass('copilot-chat-modal');

    // Chat display area
    this.chatContainer = contentEl.createDiv('chat-messages');

    // Input area
    const inputContainer = contentEl.createDiv('chat-input-container');

    new Setting(inputContainer)
      .setName('Your message')
      .addTextArea(text => {
        this.inputArea = text;
        text.setPlaceholder('Type your message...');
        text.inputEl.rows = 4;
        text.inputEl.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' && e.ctrlKey) {
            this.sendMessage();
          }
        });
      });

    new Setting(inputContainer)
      .addButton(btn => btn
        .setButtonText('Send')
        .setCta()
        .onClick(() => this.sendMessage()));
  }

  async sendMessage() {
    const userMessage = this.inputArea.getValue().trim();
    if (!userMessage) return;

    // Add user message to history and display
    this.chatHistory.push({ role: 'user', content: userMessage });
    this.displayMessage('user', userMessage);
    this.inputArea.setValue('');

    // Get AI response
    try {
      const response = await this.client.chat(this.chatHistory);
      this.chatHistory.push({ role: 'assistant', content: response });
      this.displayMessage('assistant', response);
    } catch (error) {
      this.displayMessage('error', 'Failed to get response from AI');
    }
  }

  displayMessage(role: string, content: string) {
    const msgDiv = this.chatContainer.createDiv(`chat-message chat-${role}`);
    msgDiv.createEl('strong', { text: role === 'user' ? 'You: ' : 'AI: ' });
    msgDiv.createEl('span', { text: content });
    this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
  }

  onClose() {
    this.contentEl.empty();
  }
}
```

**Critical rules**:
- Extend `Modal` for dialog UI
- Store chat history as array of messages
- Use `createDiv()` and `createEl()` for DOM manipulation
- Auto-scroll chat container on new messages
- Support Ctrl+Enter keyboard shortcut
- Clean up with `contentEl.empty()` in `onClose()`

## Step 5: Vault Integration & Context

Add vault awareness for context-aware responses:

```typescript
class CopilotChatModal extends Modal {
  // ... previous code ...

  private async getVaultContext(): Promise<string> {
    const activeFile = this.app.workspace.getActiveFile();
    if (!activeFile) return '';

    const content = await this.app.vault.read(activeFile);
    return `Current note: ${activeFile.basename}\n\n${content}`;
  }

  async sendMessage() {
    const userMessage = this.inputArea.getValue().trim();
    if (!userMessage) return;

    // Add system context on first message
    if (this.chatHistory.length === 0) {
      const context = await this.getVaultContext();
      if (context) {
        this.chatHistory.push({
          role: 'system',
          content: `You are a helpful assistant working within Obsidian. ${context}`
        });
      }
    }

    // ... rest of sendMessage implementation
  }
}
```

**Critical rules**:
- Use `app.workspace.getActiveFile()` for current note
- Read file content with `app.vault.read()`
- Add system message with context on first interaction
- Keep context minimal to avoid token limits
- Check for null/undefined before accessing vault

## Quick Reference Checklist

- [ ] Create plugin class extending `Plugin`
- [ ] Define `CopilotSettings` interface with endpoint, API key, model name
- [ ] Implement `loadSettings()` and `saveSettings()` methods
- [ ] Create `CopilotSettingTab` for configuration UI
- [ ] Build `AIClient` class with `chat()` method for API calls
- [ ] Implement OpenAI-compatible `/v1/chat/completions` endpoint
- [ ] Create `CopilotChatModal` extending `Modal` for chat UI
- [ ] Add command registration in `onload()` method
- [ ] Handle errors with try/catch and user notices
- [ ] Integrate vault context with `app.workspace` and `app.vault`
- [ ] Test with local endpoints (Ollama, LM Studio)
- [ ] Verify CORS settings for local model servers
- [ ] Add keyboard shortcuts for better UX
- [ ] Clean up resources in `onClose()` methods

## Local Model Setup Notes

**For Ollama**:
```bash
# Set CORS before starting
OLLAMA_ORIGINS="app://obsidian.md*" ollama serve

# Pull model
ollama pull llama2
```

**For LM Studio**:
1. Enable CORS in Local Server settings
2. Enable hardware acceleration
3. Start server on port 1234
4. Use endpoint: `http://localhost:1234/v1`

## Common Issues

- **CORS errors**: Set `OLLAMA_ORIGINS` or enable CORS in model server
- **401 Unauthorized**: Check API key format and header
- **Timeout**: Increase max tokens or use smaller model
- **Empty responses**: Verify endpoint URL ends with `/v1`
