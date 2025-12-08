/**
 * API Client for FracFire Backend
 * Handles all communication with the FastAPI backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export class FracFireAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `API Error: ${response.status}`);
    }

    return response.json();
  }

  // Health check
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    return this.request('/health');
  }

  // Data Generation
  async generateData(params: {
    symbol: string;
    bars: number;
    volatility: number;
    seed?: number;
  }): Promise<{
    success: boolean;
    dataset_id: string;
    bars: number;
    data: any[];
  }> {
    return this.request('/api/data/generate', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  // Dataset Management
  async listDatasets(): Promise<{ datasets: any[] }> {
    return this.request('/api/data/datasets');
  }

  async getDataset(datasetId: string): Promise<{ id: string; data: any[] }> {
    return this.request(`/api/data/dataset/${datasetId}`);
  }

  async uploadDataset(file: File): Promise<{
    success: boolean;
    dataset_id: string;
    bars: number;
  }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/api/data/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Upload failed');
    }

    return response.json();
  }

  // Model Training
  async trainModel(params: {
    dataset_id: string;
    model_type: string;
    epochs?: number;
    batch_size?: number;
  }): Promise<{
    success: boolean;
    model_id: string;
    model_type: string;
    accuracy: number;
    status: string;
  }> {
    return this.request('/api/models/train', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  async listModels(): Promise<{ models: any[] }> {
    return this.request('/api/models');
  }

  // Setup Analysis
  async analyzeSetups(params: {
    strategy_id: string;
    dataset_id: string;
    risk_reward?: number;
  }): Promise<{
    success: boolean;
    setups: any[];
  }> {
    return this.request('/api/setups/analyze', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  // Script Execution
  async listScripts(): Promise<{ scripts: any[] }> {
    return this.request('/api/scripts');
  }

  async executeScript(scriptName: string, args: Record<string, any> = {}): Promise<{
    success: boolean;
    output: string;
    script: string;
  }> {
    return this.request('/api/scripts/execute', {
      method: 'POST',
      body: JSON.stringify({ script_name: scriptName, args }),
    });
  }
}

// Export singleton instance
export const api = new FracFireAPI();
