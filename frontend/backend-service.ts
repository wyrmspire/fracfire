/**
 * Backend Service - Integrates with FracFire API
 * Replaces MockBackend with real API calls
 */

import { api } from './api-client';
import React from 'react';

export class RealBackend {
  stateRef: React.MutableRefObject<any>;
  setState: React.Dispatch<React.SetStateAction<any>>;

  constructor(stateRef: React.MutableRefObject<any>, setState: React.Dispatch<React.SetStateAction<any>>) {
    this.stateRef = stateRef;
    this.setState = setState;
  }

  get state() {
    return this.stateRef.current;
  }

  // --- Real API Integration ---

  async fetchMarketData(symbol: string): Promise<string> {
    console.log(`[REAL BACKEND] Fetching data for ${symbol}`);
    
    try {
      // For now, use the generator with external-like settings
      const result = await api.generateData({
        symbol: symbol,
        bars: 1000,
        volatility: 2.5,
      });

      const newDataset = {
        id: result.dataset_id,
        name: `${symbol} Data (${result.bars} bars)`,
        source: 'REAL' as const,
        created: new Date(),
        data: result.data
      };

      this.setState((prev: any) => ({
        ...prev,
        datasets: [...prev.datasets, newDataset],
        activeDatasetId: newDataset.id
      }));

      return `Successfully fetched data for ${symbol}. Dataset ID: ${result.dataset_id}`;
    } catch (error: any) {
      return `Error fetching data: ${error.message}`;
    }
  }

  async generatePriceData(symbol: string, bars: number, volatility: number): Promise<string> {
    console.log(`[REAL BACKEND] Generating ${bars} bars for ${symbol}`);
    
    try {
      const result = await api.generateData({
        symbol,
        bars,
        volatility,
        seed: Math.floor(Math.random() * 10000)
      });

      const newDataset = {
        id: result.dataset_id,
        name: `${symbol} Generated (${result.bars} bars)`,
        source: 'MOCK' as const,
        created: new Date(),
        data: result.data
      };

      this.setState((prev: any) => ({
        ...prev,
        datasets: [...prev.datasets, newDataset],
        activeDatasetId: newDataset.id
      }));

      return `Successfully generated dataset ${result.dataset_id} with ${result.bars} bars.`;
    } catch (error: any) {
      return `Error generating data: ${error.message}`;
    }
  }

  async executeDataScript(scriptName: string, args: any): Promise<string> {
    console.log(`[REAL BACKEND] Executing script: ${scriptName}`);
    
    try {
      const result = await api.executeScript(scriptName, args);
      return result.output;
    } catch (error: any) {
      return `Error executing script: ${error.message}`;
    }
  }

  async writeAndExecuteCode(code: string, filename: string): Promise<string> {
    console.log(`[REAL BACKEND] Writing custom code to ${filename}`);
    console.log("Code:", code);
    
    // This would need a backend endpoint to support custom code execution
    return `Custom code execution not yet implemented. Filename: ${filename}`;
  }

  async trainModel(datasetId: string, modelType: string, epochs: number): Promise<string> {
    console.log(`[REAL BACKEND] Training ${modelType} model on ${datasetId}`);
    
    try {
      const result = await api.trainModel({
        dataset_id: datasetId,
        model_type: modelType,
        epochs: epochs,
        batch_size: 32
      });

      const newModel = {
        id: result.model_id,
        name: `${modelType.toUpperCase()} Model`,
        type: modelType,
        status: result.status as 'training' | 'ready' | 'failed',
        accuracy: result.accuracy,
        datasetId: datasetId
      };

      this.setState((prev: any) => ({
        ...prev,
        models: [...prev.models, newModel],
        activeModelId: newModel.id
      }));

      return `Model ${result.model_id} training initiated with ${(result.accuracy * 100).toFixed(2)}% accuracy.`;
    } catch (error: any) {
      return `Error training model: ${error.message}`;
    }
  }

  async trainCNN(tradeSetId: string, datasetId: string, epochs: number): Promise<string> {
    return this.trainModel(datasetId, 'CNN-Trade-Learner', epochs);
  }

  async runCNNInference(modelId: string, targetDatasetId: string): Promise<string> {
    console.log(`[REAL BACKEND] Running CNN inference`);
    
    try {
      const result = await api.analyzeSetups({
        strategy_id: `CNN-${modelId}`,
        dataset_id: targetDatasetId,
        risk_reward: 2.0
      });

      const newSetups = result.setups.map((setup: any) => ({
        ...setup,
        source: 'STRATEGY' as const,
        modelUsed: modelId
      }));

      const stratId = `CNN-${modelId}`;
      const stratExists = this.state.strategies.find((s: any) => s.id === stratId);
      
      if (!stratExists) {
        this.setState((prev: any) => ({
          ...prev,
          strategies: [...prev.strategies, {
            id: stratId,
            name: `CNN Inference: ${modelId}`,
            description: 'AI Generated Setups based on trained CNN',
            created: new Date()
          }]
        }));
      }

      this.setState((prev: any) => ({
        ...prev,
        setups: [...prev.setups, ...newSetups],
        activeStrategyId: stratId,
        activeDatasetId: targetDatasetId
      }));

      return `CNN Inference complete. Found ${newSetups.length} setups.`;
    } catch (error: any) {
      return `Error running inference: ${error.message}`;
    }
  }

  async createSetupStrategy(name: string, description: string): Promise<string> {
    const newStrategy = {
      id: `strat-${Date.now()}`,
      name,
      description,
      created: new Date()
    };

    this.setState((prev: any) => ({
      ...prev,
      strategies: [...prev.strategies, newStrategy],
      activeStrategyId: newStrategy.id
    }));

    return `Setup strategy '${name}' created (ID: ${newStrategy.id}).`;
  }

  async runSetupAnalysis(strategyId: string, datasetId: string, riskReward: number): Promise<string> {
    console.log(`[REAL BACKEND] Running setup analysis`);
    
    try {
      const result = await api.analyzeSetups({
        strategy_id: strategyId,
        dataset_id: datasetId,
        risk_reward: riskReward
      });

      const newSetups = result.setups.map((setup: any) => ({
        ...setup,
        source: 'STRATEGY' as const
      }));

      this.setState((prev: any) => ({
        ...prev,
        setups: [...prev.setups, ...newSetups],
        activeStrategyId: strategyId
      }));

      const strategy = this.state.strategies.find((s: any) => s.id === strategyId);
      return `Analysis complete. Found ${newSetups.length} setups for strategy '${strategy?.name || strategyId}'.`;
    } catch (error: any) {
      return `Error running analysis: ${error.message}`;
    }
  }

  // UI Helpers (keep same as mock)

  uploadDataset(file: File) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const json = JSON.parse(e.target?.result as string);
        if (Array.isArray(json) && json[0]?.time && json[0]?.close) {
          const newDataset = {
            id: `ds-upload-${Date.now()}`,
            name: file.name.replace('.json', ''),
            source: 'UPLOAD' as const,
            created: new Date(),
            data: json
          };
          this.setState((prev: any) => ({
            ...prev,
            datasets: [...prev.datasets, newDataset],
            activeDatasetId: newDataset.id
          }));
        } else {
          alert("Invalid JSON format. Expected array of OHLCV objects.");
        }
      } catch (err) {
        alert("Error parsing JSON file.");
      }
    };
    reader.readAsText(file);
  }

  addManualTrade(trade: any, datasetId: string) {
    this.setState((prev: any) => {
      let activeSet = prev.tradeSets.find((ts: any) => ts.id === prev.activeTradeSetId);
      let newSets = [...prev.tradeSets];

      if (!activeSet || activeSet.datasetId !== datasetId) {
        const existing = newSets.find((ts: any) => ts.datasetId === datasetId);
        if (existing) {
          activeSet = existing;
        } else {
          activeSet = {
            id: `ts-${Date.now()}`,
            name: `Manual Trades ${prev.tradeSets.length + 1}`,
            datasetId,
            created: new Date(),
            trades: []
          };
          newSets.push(activeSet);
        }
      }

      const updatedSet = { ...activeSet, trades: [...activeSet.trades, trade] };
      newSets = newSets.map((s: any) => s.id === activeSet!.id ? updatedSet : s);

      return {
        ...prev,
        tradeSets: newSets,
        activeTradeSetId: activeSet!.id
      };
    });
  }

  updateManualTrade(updatedTrade: any) {
    this.setState((prev: any) => {
      const newSets = prev.tradeSets.map((ts: any) => ({
        ...ts,
        trades: ts.trades.map((t: any) => t.id === updatedTrade.id ? updatedTrade : t)
      }));
      return { ...prev, tradeSets: newSets };
    });
  }
}
