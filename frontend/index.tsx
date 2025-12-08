import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Type, FunctionDeclaration } from "@google/genai";
import { RealBackend } from './backend-service';
import './index.css';

console.log("Initializing FracFire Studio...");

/**
 * QUANT AI STUDIO
 * 
 * A React 18 application for financial modeling, data generation, and strategy development.
 * Integrated with Google Gemini 3 Pro for conversational agent capabilities.
 */

// =============================================================================
// 1. TYPE DEFINITIONS & DOMAIN MODELS
// =============================================================================

type OHLCV = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  original_symbol: string;
};

type StrategyStats = {
  totalTrades: number;
  winRate: number;
  profitFactor: number;
  netPnL: number;
  maxDrawdown: number;
};

type SetupStrategy = {
  id: string;
  name: string;
  description: string;
  created: Date;
  stats?: StrategyStats; // Cached stats from last run
};

type TradeSetup = {
  id: string;
  strategyId: string; // If 'MANUAL', it belongs to a manual set
  time: string;
  type: 'LONG' | 'SHORT';
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;
  confidence: number;
  modelUsed: string;
  source: 'STRATEGY' | 'MANUAL'; // Distinguished source
  manualExitTime?: string; // Optional override for manual resizing
};

type TradeSet = {
  id: string;
  name: string;
  datasetId: string;
  created: Date;
  trades: TradeSetup[];
};

type ProcessedSetup = TradeSetup & {
  startIndex: number;
  endIndex: number;
  outcome: 'TP' | 'SL' | 'OPEN' | 'MANUAL_EXIT';
  endTime: string;
  pnl: number;
};

type Dataset = {
  id: string;
  name: string;
  source: 'MOCK' | 'REAL' | 'UPLOAD';
  created: Date;
  data: OHLCV[];
};

type MLModel = {
  id: string;
  name: string;
  type: string;
  status: 'training' | 'ready' | 'failed';
  accuracy: number;
  datasetId: string; // The dataset (or trade set) it was trained on
};

type AppState = {
  datasets: Dataset[];
  models: MLModel[];
  strategies: SetupStrategy[];
  tradeSets: TradeSet[]; // Manual trade collections
  activeDatasetId: string | null;
  activeModelId: string | null;
  activeStrategyId: string | null;
  activeTradeSetId: string | null;
  setups: TradeSetup[]; // Strategy-generated setups
  chartTimeframe: number; // in minutes (1, 5, 15, 60)
};

// =============================================================================
// 2. DATA UTILITIES (Resampling, Stats & Generation)
// =============================================================================

// Helper for Tick Size (0.25)
const TICK_SIZE = 0.25;
const roundToTick = (value: number) => {
    return Math.round(value / TICK_SIZE) * TICK_SIZE;
};

const SEED_DATA: OHLCV[] = [
  { "time": "2025-03-18T00:00:00Z", "open": 5810.25, "high": 5811.5, "low": 5809.75, "close": 5810.0, "volume": 608, "original_symbol": "MESM5" },
  { "time": "2025-03-18T00:01:00Z", "open": 5810.25, "high": 5812.75, "low": 5810.0, "close": 5812.0, "volume": 237, "original_symbol": "MESM5" },
  { "time": "2025-03-18T00:02:00Z", "open": 5811.5, "high": 5812.0, "low": 5811.5, "close": 5812.0, "volume": 35, "original_symbol": "MESM5" },
  { "time": "2025-03-18T00:03:00Z", "open": 5811.5, "high": 5812.5, "low": 5811.25, "close": 5811.75, "volume": 246, "original_symbol": "MESM5" },
  { "time": "2025-03-18T00:04:00Z", "open": 5811.75, "high": 5812.5, "low": 5811.75, "close": 5812.5, "volume": 57, "original_symbol": "MESM5" },
  { "time": "2025-03-18T00:05:00Z", "open": 5812.25, "high": 5812.5, "low": 5811.0, "close": 5811.25, "volume": 74, "original_symbol": "MESM5" },
  { "time": "2025-03-18T00:06:00Z", "open": 5811.0, "high": 5811.0, "low": 5809.25, "close": 5810.25, "volume": 221, "original_symbol": "MESM5" },
  { "time": "2025-03-18T00:07:00Z", "open": 5810.0, "high": 5811.25, "low": 5809.5, "close": 5811.0, "volume": 170, "original_symbol": "MESM5" },
  { "time": "2025-03-18T00:08:00Z", "open": 5810.25, "high": 5810.25, "low": 5809.0, "close": 5809.75, "volume": 198, "original_symbol": "MESM5" },
  { "time": "2025-03-18T00:09:00Z", "open": 5809.75, "high": 5810.5, "low": 5809.25, "close": 5810.0, "volume": 162, "original_symbol": "MESM5" },
  { "time": "2025-03-18T00:10:00Z", "open": 5810.0, "high": 5810.25, "low": 5809.5, "close": 5810.0, "volume": 74, "original_symbol": "MESM5" },
  { "time": "2025-03-18T00:11:00Z", "open": 5810.25, "high": 5811.5, "low": 5808.5, "close": 5811.0, "volume": 288, "original_symbol": "MESM5" },
  { "time": "2025-03-18T00:12:00Z", "open": 5810.75, "high": 5811.5, "low": 5810.25, "close": 5810.5, "volume": 217, "original_symbol": "MESM5" },
  { "time": "2025-03-18T00:13:00Z", "open": 5810.75, "high": 5811.5, "low": 5810.5, "close": 5810.75, "volume": 84, "original_symbol": "MESM5" },
  { "time": "2025-03-18T00:14:00Z", "open": 5810.75, "high": 5811.5, "low": 5810.75, "close": 5810.75, "volume": 47, "original_symbol": "MESM5" },
];

const generateMockData = (symbol: string, count: number, volatility: number): OHLCV[] => {
  let currentPrice = 5810.0;
  const data: OHLCV[] = [];
  
  // Align start time to nearest minute to avoid ms issues causing findIndex to fail
  const now = new Date();
  now.setSeconds(0, 0);
  const startTime = now.getTime() - (count * 60000);

  for (let i = 0; i < count; i++) {
    const change = (Math.random() - 0.5) * volatility;
    let open = currentPrice;
    let close = open + change;
    let high = Math.max(open, close) + Math.random() * (volatility / 2);
    let low = Math.min(open, close) - Math.random() * (volatility / 2);

    // Enforce Tick Size
    open = roundToTick(open);
    close = roundToTick(close);
    high = roundToTick(high);
    low = roundToTick(low);

    // Sanity check candles
    if (high < Math.max(open, close)) high = Math.max(open, close);
    if (low > Math.min(open, close)) low = Math.min(open, close);

    const volume = Math.floor(Math.random() * 500) + 50;
    
    currentPrice = close;
    
    data.push({
      time: new Date(startTime + i * 60000).toISOString(),
      open,
      high,
      low,
      close,
      volume,
      original_symbol: symbol
    });
  }
  return data;
};

// Resamples 1m data into higher timeframes
const resampleData = (data: OHLCV[], timeframeMinutes: number): OHLCV[] => {
  if (timeframeMinutes === 1) return data;
  
  const resampled: OHLCV[] = [];
  let currentBar: OHLCV | null = null;
  let barStartTime = 0;

  for (const candle of data) {
    const time = new Date(candle.time).getTime();
    // Align time to bucket
    const bucketTime = Math.floor(time / (timeframeMinutes * 60000)) * (timeframeMinutes * 60000);

    if (currentBar && bucketTime !== barStartTime) {
      resampled.push(currentBar);
      currentBar = null;
    }

    if (!currentBar) {
      barStartTime = bucketTime;
      currentBar = {
        time: new Date(bucketTime).toISOString(),
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        volume: candle.volume,
        original_symbol: candle.original_symbol
      };
    } else {
      currentBar.high = Math.max(currentBar.high, candle.high);
      currentBar.low = Math.min(currentBar.low, candle.low);
      currentBar.close = candle.close; // Close is always the latest close
      currentBar.volume += candle.volume;
    }
  }
  if (currentBar) resampled.push(currentBar);

  return resampled;
};

// =============================================================================
// 3. BACKEND SERVICE LAYER
// =============================================================================

// RealBackend has been moved to backend-service.ts as RealBackend
// The RealBackend connects to the FastAPI backend at localhost:8000

// =============================================================================
// 4. CHARTING ENGINE (React + Canvas)
// =============================================================================

const CanvasChart = ({ 
    data, 
    setups, 
    manualSets,
    activeStrategyId, 
    activeTradeSetId,
    timeframe, 
    onAddTrade,
    onUpdateTrade
}: { 
    data: OHLCV[], 
    setups: TradeSetup[], 
    manualSets: TradeSet[],
    activeStrategyId: string | null, 
    activeTradeSetId: string | null,
    timeframe: number,
    onAddTrade: (t: TradeSetup) => void,
    onUpdateTrade: (t: TradeSetup) => void
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Interaction State
  const [crosshair, setCrosshair] = useState<{x: number, y: number} | null>(null);
  const [drawingMode, setDrawingMode] = useState<'NONE' | 'LONG' | 'SHORT'>('NONE');
  
  // Y-Axis Scale State
  const [scaleY, setScaleY] = useState(1.0);
  const [offsetY, setOffsetY] = useState(0);

  // Dragging State
  const dragRef = useRef<{
    isDragging: boolean;
    type: 'PAN' | 'Y_SCALE' | 'TRADE_BODY' | 'TRADE_TP' | 'TRADE_SL' | 'TRADE_RIGHT_EDGE';
    startX: number;
    startY: number;
    initialScaleY: number;
    initialStartIndex: number;
    targetTradeId?: string;
    tradeSnapshot?: TradeSetup;
  }>({
    isDragging: false,
    type: 'PAN',
    startX: 0,
    startY: 0,
    initialScaleY: 1,
    initialStartIndex: 0
  });

  // Hover state for cursor
  const [hoverTarget, setHoverTarget] = useState<'NONE' | 'TRADE_BODY' | 'TRADE_TP' | 'TRADE_SL' | 'TRADE_RIGHT_EDGE'>('NONE');

  // Viewport
  const [startIndex, setStartIndex] = useState(0);
  const [barWidth, setBarWidth] = useState(10);
  
  // Resampled Data
  const displayData = useMemo(() => resampleData(data, timeframe), [data, timeframe]);

  // Merge Setups
  const allSetupsToDisplay = useMemo(() => {
      let result: TradeSetup[] = [];
      if (activeStrategyId) {
          result = result.concat(setups.filter(s => s.strategyId === activeStrategyId));
      }
      if (activeTradeSetId) {
          const set = manualSets.find(s => s.id === activeTradeSetId);
          if (set) result = result.concat(set.trades);
      }
      return result;
  }, [setups, manualSets, activeStrategyId, activeTradeSetId]);

  // Process Setups (Calculate outcomes)
  const processedSetups = useMemo(() => {
    if (allSetupsToDisplay.length === 0 || data.length === 0) return [];
    
    // We process against raw 1m data for accuracy
    return allSetupsToDisplay.map(setup => {
       // Robust find: if exact match fails (e.g. resampled bucket time), find closest subsequent 1m candle
       let rawStartIndex = data.findIndex(d => d.time === setup.time);
       if (rawStartIndex === -1) {
           const setupTimeVal = new Date(setup.time).getTime();
           rawStartIndex = data.findIndex(d => new Date(d.time).getTime() >= setupTimeVal);
       }
       
       if (rawStartIndex === -1) return null;

       let rawEndIndex = data.length - 1;
       let outcome: 'TP' | 'SL' | 'OPEN' | 'MANUAL_EXIT' = 'OPEN';
       let pnl = 0;

       // If manual exit is set, we respect it strictly
       if (setup.manualExitTime) {
           let exitIdx = data.findIndex(d => d.time === setup.manualExitTime);
           if (exitIdx === -1) {
               // Fallback if exit time not found exactly in 1m data
               const exitTimeVal = new Date(setup.manualExitTime).getTime();
               exitIdx = data.findIndex(d => new Date(d.time).getTime() >= exitTimeVal);
           }

           if (exitIdx !== -1 && exitIdx > rawStartIndex) {
               rawEndIndex = exitIdx;
               outcome = 'MANUAL_EXIT';
           } else {
               // Default length if lookup fails
               rawEndIndex = Math.min(data.length - 1, rawStartIndex + 20);
           }
       } else {
           // Simulation
           for (let i = rawStartIndex; i < data.length; i++) {
               const bar = data[i];
               if (setup.type === 'LONG') {
                   if (bar.low <= setup.stopLoss) { rawEndIndex = i; outcome = 'SL'; break; }
                   if (bar.high >= setup.takeProfit) { rawEndIndex = i; outcome = 'TP'; break; }
               } else {
                   if (bar.high >= setup.stopLoss) { rawEndIndex = i; outcome = 'SL'; break; }
                   if (bar.low <= setup.takeProfit) { rawEndIndex = i; outcome = 'TP'; break; }
               }
           }
       }
       
       if (outcome === 'OPEN' && !setup.manualExitTime) rawEndIndex = data.length - 1;
       
       return { ...setup, outcome, endTime: data[rawEndIndex].time, pnl };
    }).filter(s => s !== null) as ProcessedSetup[];
  }, [allSetupsToDisplay, data]);

  // Auto-scroll to end on data load
  useEffect(() => {
    if (displayData.length > 0) {
      setStartIndex(Math.max(0, displayData.length - 100));
    }
  }, [displayData.length]);

  // Handle Wheel Zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
      e.preventDefault();
      // Zoom X-axis by changing barWidth
      const delta = Math.sign(e.deltaY) * -1;
      const zoomFactor = 0.1;
      const newWidth = Math.max(2, Math.min(200, barWidth * (1 + delta * zoomFactor)));
      setBarWidth(newWidth);
  }, [barWidth]);

  // --- DRAWING LOGIC ---
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !displayData || displayData.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const rightMargin = 60;
    const bottomMargin = 30;
    const chartW = width - rightMargin;
    const chartH = height - bottomMargin;

    // Clear
    ctx.fillStyle = '#0f1115';
    ctx.fillRect(0, 0, width, height);

    const visibleBarsCount = Math.ceil(chartW / barWidth);
    const endIndex = Math.min(startIndex + visibleBarsCount, displayData.length);
    const visibleBars = displayData.slice(startIndex, endIndex);

    if (visibleBars.length === 0) return;

    // Calculate Price Range
    let minPrice = Infinity;
    let maxPrice = -Infinity;
    visibleBars.forEach(d => {
      if (d.low < minPrice) minPrice = d.low;
      if (d.high > maxPrice) maxPrice = d.high;
    });

    const padding = (maxPrice - minPrice) * 0.1;
    let baseMin = minPrice - padding;
    let baseMax = maxPrice + padding;
    let range = baseMax - baseMin;

    // Apply Vertical Scaling/Offset
    const mid = baseMin + range / 2;
    range /= scaleY; // Zoom
    const currentMin = mid - range / 2 + offsetY;
    const currentMax = mid + range / 2 + offsetY;

    const priceToY = (price: number) => chartH - ((price - currentMin) / (currentMax - currentMin)) * chartH; 
    const yToPrice = (y: number) => currentMin + ((chartH - y) / chartH) * (currentMax - currentMin);

    // Helpers for X-axis
    const getX = (idx: number) => (idx - startIndex) * barWidth + (barWidth / 2);
    
    // Accurate Time Mapping: Finds the bar on CURRENT timeframe that contains the timestamp
    const getTimeX = (timeStr: string) => {
        const t = new Date(timeStr).getTime();
        let idx = -1;
        
        // Linear scan for bar that covers the time
        // Since displayData is sorted, we find the first bar where time > t, then back up one
        // OR find the bar that matches exactly.
        for (let i = 0; i < displayData.length; i++) {
             const barTime = new Date(displayData[i].time).getTime();
             if (barTime === t) {
                 idx = i;
                 break;
             }
             if (barTime > t) {
                 // The trade time 't' is before this bar's start. 
                 // So it must be in the previous bar (if it exists)
                 idx = i > 0 ? i - 1 : 0;
                 break;
             }
        }
        
        // Edge case: Time is after the last bar
        if (idx === -1 && displayData.length > 0) {
            const lastBarTime = new Date(displayData[displayData.length - 1].time).getTime();
            if (t >= lastBarTime) {
                idx = displayData.length - 1;
            }
        }

        if (idx === -1) return -1000;
        return getX(idx);
    };

    // Grid
    ctx.strokeStyle = '#2a2e39';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const y = (chartH / 10) * i;
      ctx.moveTo(0, y);
      ctx.lineTo(chartW, y);
      const p = yToPrice(y);
      ctx.fillStyle = '#6b7280';
      ctx.fillText(p.toFixed(2), chartW + 5, y + 3);
    }
    ctx.stroke();

    // Candles
    const gap = Math.max(1, Math.floor(barWidth * 0.2));
    const effectiveBarWidth = Math.max(1, barWidth - gap);

    visibleBars.forEach((d, i) => {
        const x = getX(startIndex + i);
        const openY = priceToY(d.open);
        const closeY = priceToY(d.close);
        const highY = priceToY(d.high);
        const lowY = priceToY(d.low);
        const isUp = d.close >= d.open;
        
        ctx.fillStyle = isUp ? '#26a69a' : '#ef5350';
        ctx.strokeStyle = isUp ? '#26a69a' : '#ef5350';
        
        ctx.beginPath();
        ctx.moveTo(x, highY);
        ctx.lineTo(x, lowY);
        ctx.stroke();
        
        const h = Math.max(1, Math.abs(closeY - openY));
        const top = Math.min(openY, closeY);
        ctx.fillRect(x - effectiveBarWidth / 2, top, effectiveBarWidth, h);

        // Time Label
        // Adaptive time labels based on density
        const labelStep = Math.max(1, Math.floor(100 / barWidth));
        if ((startIndex + i) % labelStep === 0) {
            ctx.fillStyle = '#6b7280';
            ctx.fillText(new Date(d.time).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}), x - 15, chartH + 15);
        }
    });

    // Draw Setups
    processedSetups.forEach(setup => {
        const xStart = getTimeX(setup.time);
        const xEnd = getTimeX(setup.endTime);
        const entryY = priceToY(setup.entryPrice);
        const slY = priceToY(setup.stopLoss);
        const tpY = priceToY(setup.takeProfit);

        if (xEnd < 0 || xStart > chartW) return;

        let tpColor = 'rgba(38, 166, 154, 0.15)'; 
        let slColor = 'rgba(239, 83, 80, 0.15)';
        if (setup.outcome === 'TP') tpColor = 'rgba(38, 166, 154, 0.4)';
        if (setup.outcome === 'SL') slColor = 'rgba(239, 83, 80, 0.4)';

        // Ensure width is at least 1 bar
        const widthPx = Math.max(effectiveBarWidth, xEnd - xStart);
        
        // Manual Trade Visuals (Handles)
        if (setup.source === 'MANUAL') {
             ctx.strokeStyle = '#fff';
             ctx.setLineDash([2, 2]);
             ctx.lineWidth = 1;
             
             // TP Box
             ctx.fillStyle = tpColor;
             ctx.fillRect(xStart, Math.min(entryY, tpY), widthPx, Math.abs(entryY - tpY));
             ctx.strokeRect(xStart, Math.min(entryY, tpY), widthPx, Math.abs(entryY - tpY));

             // SL Box
             ctx.fillStyle = slColor;
             ctx.fillRect(xStart, Math.min(entryY, slY), widthPx, Math.abs(entryY - slY));
             ctx.strokeRect(xStart, Math.min(entryY, slY), widthPx, Math.abs(entryY - slY));
             
             // Right Edge Drag Handle - Slimmer
             ctx.fillStyle = '#fff';
             ctx.fillRect(xStart + widthPx - 2, Math.min(tpY, slY), 4, Math.abs(tpY - slY)); 

             ctx.setLineDash([]);

             // Text Labels
             ctx.fillStyle = '#fff';
             ctx.font = '10px sans-serif';
             ctx.fillText('TP', xStart + 2, tpY - 2);
             ctx.fillText('SL', xStart + 2, slY + 10);
        } else {
             ctx.fillStyle = tpColor;
             ctx.fillRect(xStart, Math.min(entryY, tpY), widthPx, Math.abs(entryY - tpY));
             ctx.fillStyle = slColor;
             ctx.fillRect(xStart, Math.min(entryY, slY), widthPx, Math.abs(entryY - slY));
        }
    });

    // Crosshair
    if (crosshair && crosshair.x < chartW && crosshair.y < chartH) {
        ctx.strokeStyle = '#fff';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(crosshair.x, 0);
        ctx.lineTo(crosshair.x, chartH);
        ctx.moveTo(0, crosshair.y);
        ctx.lineTo(chartW, crosshair.y);
        ctx.stroke();
        ctx.setLineDash([]);
        
        const price = yToPrice(crosshair.y);
        ctx.fillStyle = '#2962ff';
        ctx.fillRect(chartW, crosshair.y - 10, rightMargin, 20);
        ctx.fillStyle = '#fff';
        ctx.fillText(price.toFixed(2), chartW + 5, crosshair.y + 4);
    }

  }, [displayData, startIndex, barWidth, processedSetups, scaleY, offsetY, crosshair]);

  // Handle Resize
  useEffect(() => {
    const handleResize = () => {
        if (containerRef.current && canvasRef.current) {
            canvasRef.current.width = containerRef.current.clientWidth;
            canvasRef.current.height = containerRef.current.clientHeight;
            draw();
        }
    };
    window.addEventListener('resize', handleResize);
    handleResize();
    return () => window.removeEventListener('resize', handleResize);
  }, [draw]);

  // --- MOUSE EVENTS ---
  
  const getChartMetrics = () => {
     if (!canvasRef.current) return null;
     const h = canvasRef.current.height - 30;
     const w = canvasRef.current.width - 60;
     
     const visibleBarsCount = Math.ceil(w / barWidth);
     const endIndex = Math.min(startIndex + visibleBarsCount, displayData.length);
     const visibleBars = displayData.slice(startIndex, endIndex);
     if(visibleBars.length === 0) return null;

     let minPrice = Infinity; let maxPrice = -Infinity;
     visibleBars.forEach(d => { if (d.low < minPrice) minPrice = d.low; if (d.high > maxPrice) maxPrice = d.high; });
     const padding = (maxPrice - minPrice) * 0.1;
     let baseMin = minPrice - padding;
     let baseMax = maxPrice + padding;
     let range = (baseMax - baseMin) / scaleY;
     const mid = baseMin + (baseMax - baseMin) / 2;
     const currentMin = mid - range / 2 + offsetY;
     const currentMax = mid + range / 2 + offsetY;
     
     return { chartH: h, chartW: w, minP: currentMin, maxP: currentMax };
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const metrics = getChartMetrics();
    if (!metrics) return;
    
    // 1. Check Y-Axis Drag
    if (x > metrics.chartW) {
        dragRef.current = {
            isDragging: true,
            type: 'Y_SCALE',
            startX: x,
            startY: y,
            initialScaleY: scaleY,
            initialStartIndex: startIndex
        };
        return;
    }

    // 2. Check Hit Test on Trades (Only Manual Trades are interactive)
    if (drawingMode === 'NONE') {
        const pricePerY = (metrics.maxP - metrics.minP) / metrics.chartH;
        const clickPrice = metrics.maxP - (y / metrics.chartH) * (metrics.maxP - metrics.minP);
        
        for (const trade of processedSetups) {
            if (trade.source !== 'MANUAL') continue;
            
            // Map coordinates for hit testing using same logic as Draw
            let idx = -1;
            const t = new Date(trade.time).getTime();
            for (let i = 0; i < displayData.length; i++) {
                const barTime = new Date(displayData[i].time).getTime();
                if (barTime === t) { idx = i; break; }
                if (barTime > t) { idx = i > 0 ? i - 1 : 0; break; }
            }
            if (idx === -1 && displayData.length > 0) {
               if (t >= new Date(displayData[displayData.length-1].time).getTime()) idx = displayData.length - 1;
            }

            if (idx === -1) continue;

            const xStart = (idx - startIndex) * barWidth + (barWidth / 2);
            
            // For End time
            let endIdx = -1;
            const tEnd = new Date(trade.endTime).getTime();
            for (let i = 0; i < displayData.length; i++) {
                const barTime = new Date(displayData[i].time).getTime();
                if (barTime === tEnd) { endIdx = i; break; }
                if (barTime > tEnd) { endIdx = i > 0 ? i - 1 : 0; break; }
            }
             if (endIdx === -1 && displayData.length > 0) {
               if (tEnd >= new Date(displayData[displayData.length-1].time).getTime()) endIdx = displayData.length - 1;
            }
            
            // If mapping fails, fallback to simple offset but robust
            const xEnd = (endIdx !== -1) ? (endIdx - startIndex) * barWidth + (barWidth / 2) : xStart + barWidth;
            
            const width = Math.max(barWidth, xEnd - xStart);
            
            if (x >= xStart && x <= xStart + width) {
                // Check RIGHT EDGE (Resize)
                if (Math.abs(x - (xStart + width)) < 8) { // Comfortable hit area, slim visual
                    dragRef.current = { isDragging: true, type: 'TRADE_RIGHT_EDGE', startX: x, startY: y, initialScaleY: 1, initialStartIndex: 0, targetTradeId: trade.id, tradeSnapshot: trade };
                    return;
                }

                // Check Y bounds for TP/SL
                const tolerancePrice = pricePerY * 5; 
                
                if (Math.abs(clickPrice - trade.takeProfit) < tolerancePrice) {
                    dragRef.current = { isDragging: true, type: 'TRADE_TP', startX: x, startY: y, initialScaleY: 1, initialStartIndex: 0, targetTradeId: trade.id, tradeSnapshot: trade };
                    return;
                }
                if (Math.abs(clickPrice - trade.stopLoss) < tolerancePrice) {
                    dragRef.current = { isDragging: true, type: 'TRADE_SL', startX: x, startY: y, initialScaleY: 1, initialStartIndex: 0, targetTradeId: trade.id, tradeSnapshot: trade };
                    return;
                }
                
                const top = Math.max(trade.entryPrice, Math.max(trade.takeProfit, trade.stopLoss));
                const bottom = Math.min(trade.entryPrice, Math.min(trade.takeProfit, trade.stopLoss));
                
                if (clickPrice < top && clickPrice > bottom) {
                    dragRef.current = { isDragging: true, type: 'TRADE_BODY', startX: x, startY: y, initialScaleY: 1, initialStartIndex: 0, targetTradeId: trade.id, tradeSnapshot: trade };
                    return;
                }
            }
        }
    }

    // 3. Default: Pan Chart or Place Trade
    if (drawingMode !== 'NONE') {
         const barIdx = Math.floor(x / barWidth) + startIndex;
         if (displayData[barIdx]) {
            const bar = displayData[barIdx];
            const time = bar.time;
            
            // Calculate default exit time (e.g., 15 bars ahead) for fixed width
            const exitBarIdx = Math.min(displayData.length - 1, barIdx + 15);
            const exitTime = displayData[exitBarIdx].time;

            const isLong = drawingMode === 'LONG';
            
            // SNAP TO CLOSE
            const entryPrice = roundToTick(bar.close);

            // DYNAMIC RISK/REWARD SIZE based on candle size
            // Use 2x the candle range, but clamped to reasonable limits
            const candleRange = Math.max(TICK_SIZE * 4, (bar.high - bar.low));
            // Max initial size: 20 points (80 ticks) or similar, to keep it on screen
            const initialDist = Math.min(roundToTick(candleRange * 2), 20.0);

            const slPrice = roundToTick(isLong ? entryPrice - initialDist : entryPrice + initialDist);
            const tpPrice = roundToTick(isLong ? entryPrice + initialDist : entryPrice - initialDist);
            
            onAddTrade({
                id: `manual-${Date.now()}`,
                strategyId: 'MANUAL',
                source: 'MANUAL',
                time,
                type: isLong ? 'LONG' : 'SHORT',
                entryPrice: entryPrice,
                stopLoss: slPrice,
                takeProfit: tpPrice,
                confidence: 1.0,
                modelUsed: 'Manual',
                manualExitTime: exitTime // Initialize with fixed duration
            });
            setDrawingMode('NONE');
         }
    } else {
        dragRef.current = {
            isDragging: true,
            type: 'PAN',
            startX: e.clientX,
            startY: e.clientY,
            initialScaleY: 1,
            initialStartIndex: startIndex
        };
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect) {
        setCrosshair({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    }

    // Update Hover Cursor
    if (!dragRef.current.isDragging && drawingMode === 'NONE' && rect) {
        const metrics = getChartMetrics();
        if (metrics) {
            const y = e.clientY - rect.top;
            const x = e.clientX - rect.left;
            const clickPrice = metrics.maxP - (y / metrics.chartH) * (metrics.maxP - metrics.minP);
            const pricePerY = (metrics.maxP - metrics.minP) / metrics.chartH;
            const tolerance = pricePerY * 5;

            let cursorSet = false;
            for (const trade of processedSetups) {
                if (trade.source !== 'MANUAL') continue;
                
                // Use same mapping as Draw logic for hover test
                let idx = -1;
                const t = new Date(trade.time).getTime();
                for(let i=0; i<displayData.length; i++) {
                    const bt = new Date(displayData[i].time).getTime();
                    if(bt === t) { idx = i; break; }
                    if(bt > t) { idx = i>0?i-1:0; break; }
                }
                if(idx === -1 && displayData.length > 0 && t >= new Date(displayData[displayData.length-1].time).getTime()) idx = displayData.length-1;

                if (idx === -1) continue;

                const xStart = (idx - startIndex) * barWidth + (barWidth / 2);
                
                // End Mapping
                let endIdx = -1;
                const tEnd = new Date(trade.endTime).getTime();
                for(let i=0; i<displayData.length; i++) {
                    const bt = new Date(displayData[i].time).getTime();
                    if(bt === tEnd) { endIdx = i; break; }
                    if(bt > tEnd) { endIdx = i>0?i-1:0; break; }
                }
                if(endIdx === -1 && displayData.length > 0 && tEnd >= new Date(displayData[displayData.length-1].time).getTime()) endIdx = displayData.length-1;
                
                const xEnd = (endIdx !== -1) ? (endIdx - startIndex) * barWidth + (barWidth / 2) : xStart + barWidth;
                const w = Math.max(barWidth, xEnd - xStart);
                
                if (x >= xStart && x <= xStart + w) {
                    if (Math.abs(x - (xStart + w)) < 8) {
                        setHoverTarget('TRADE_RIGHT_EDGE');
                        cursorSet = true;
                        break;
                    }

                    if (Math.abs(clickPrice - trade.takeProfit) < tolerance || Math.abs(clickPrice - trade.stopLoss) < tolerance) {
                        setHoverTarget('TRADE_TP'); // Reusing handler logic (TP and SL are vertical resizing)
                        cursorSet = true;
                        break;
                    }
                    const top = Math.max(trade.entryPrice, Math.max(trade.takeProfit, trade.stopLoss));
                    const bottom = Math.min(trade.entryPrice, Math.min(trade.takeProfit, trade.stopLoss));
                    if (clickPrice < top && clickPrice > bottom) {
                        setHoverTarget('TRADE_BODY');
                        cursorSet = true;
                        break;
                    }
                }
            }
            if (!cursorSet) setHoverTarget('NONE');
        }
    }

    if (!dragRef.current.isDragging) return;

    // Handle Drag Actions
    const dy = e.clientY - (rect?.top || 0) - dragRef.current.startY;
    const dxGlobal = e.clientX - dragRef.current.startX; 

    if (dragRef.current.type === 'PAN') {
        const barsMoved = Math.round(dxGlobal / barWidth);
        setStartIndex(Math.max(0, dragRef.current.initialStartIndex - barsMoved));
        return;
    }

    if (dragRef.current.type === 'Y_SCALE') {
        const sensitivity = 0.005;
        const newScale = Math.max(0.1, Math.min(10, dragRef.current.initialScaleY * (1 - dy * sensitivity)));
        setScaleY(newScale);
        return;
    }

    // Trade Interactions
    if (['TRADE_BODY', 'TRADE_SL', 'TRADE_TP', 'TRADE_RIGHT_EDGE'].includes(dragRef.current.type)) {
        const metrics = getChartMetrics();
        if (!metrics || !dragRef.current.tradeSnapshot) return;

        const snapshot = dragRef.current.tradeSnapshot;
        const pricePerPixel = (metrics.maxP - metrics.minP) / metrics.chartH;
        const priceDelta = -(e.clientY - rect!.top - dragRef.current.startY) * pricePerPixel;
        
        let updatedTrade = { ...snapshot };

        if (dragRef.current.type === 'TRADE_RIGHT_EDGE') {
             // Calculate Time Delta based on mouse X
             const x = e.clientX - rect!.left;
             const barIdx = Math.floor(x / barWidth) + startIndex;
             // Ensure it's within bounds and after entry
             if (displayData[barIdx]) {
                 const newExitTime = displayData[barIdx].time;
                 const entryIdx = displayData.findIndex(d => d.time === snapshot.time);
                 // Only update if exit is after entry or logic permits
                 // (Note: snapshot.time might not be in displayData exactly, but we use strict compare here for simplicity in drag)
                 // A better approach is comparing Timestamps
                 const entryTimeVal = new Date(snapshot.time).getTime();
                 const newExitTimeVal = new Date(newExitTime).getTime();
                 
                 if (newExitTimeVal > entryTimeVal) {
                    updatedTrade.manualExitTime = newExitTime;
                 }
             }
        }
        else if (dragRef.current.type === 'TRADE_BODY') {
            const snappedDelta = roundToTick(priceDelta);
            updatedTrade.entryPrice = roundToTick(snapshot.entryPrice + snappedDelta);
            updatedTrade.stopLoss = roundToTick(snapshot.stopLoss + snappedDelta);
            updatedTrade.takeProfit = roundToTick(snapshot.takeProfit + snappedDelta);
            
            // X-Axis Move (Time Shift)
            const x = e.clientX - rect!.left;
            const startX = dragRef.current.startX;
            const barsShift = Math.round((x - startX) / barWidth);
            
            if (barsShift !== 0) {
                // Find current bar index in display data
                // We use fuzzy search to find where the trade 'is' on this TF
                let currentIdx = -1;
                const t = new Date(snapshot.time).getTime();
                for(let i=0; i<displayData.length; i++) {
                    const bt = new Date(displayData[i].time).getTime();
                    if(bt >= t) { currentIdx = i; break; }
                }
                
                if (currentIdx !== -1) {
                     const newIdx = Math.min(displayData.length - 1, Math.max(0, currentIdx + barsShift));
                     updatedTrade.time = displayData[newIdx].time;
                     
                     // Shift exit time by same amount of bars if possible
                     if (snapshot.manualExitTime) {
                         let currentExitIdx = -1;
                         const tExit = new Date(snapshot.manualExitTime).getTime();
                         for(let i=0; i<displayData.length; i++) {
                             const bt = new Date(displayData[i].time).getTime();
                             if(bt >= tExit) { currentExitIdx = i; break; }
                         }
                         if (currentExitIdx !== -1) {
                             const newExitIdx = Math.min(displayData.length - 1, currentExitIdx + barsShift);
                             updatedTrade.manualExitTime = displayData[newExitIdx].time;
                         }
                     }
                }
            }
        } 
        else if (dragRef.current.type === 'TRADE_SL') {
            updatedTrade.stopLoss = roundToTick(snapshot.stopLoss + priceDelta);
        }
        else if (dragRef.current.type === 'TRADE_TP') {
            updatedTrade.takeProfit = roundToTick(snapshot.takeProfit + priceDelta);
        }

        onUpdateTrade(updatedTrade);
    }
  };

  const handleMouseUp = () => {
    dragRef.current.isDragging = false;
  };

  // Cursor style
  const cursorStyle = useMemo(() => {
     if (drawingMode !== 'NONE') return 'cursor-cell';
     if (hoverTarget === 'TRADE_TP' || hoverTarget === 'TRADE_SL') return 'cursor-ns-resize';
     if (hoverTarget === 'TRADE_BODY') return 'cursor-move';
     if (hoverTarget === 'TRADE_RIGHT_EDGE') return 'cursor-ew-resize';
     return 'cursor-crosshair';
  }, [drawingMode, hoverTarget]);

  return (
    <div ref={containerRef} className={`w-full h-full relative overflow-hidden bg-[#0f1115] ${cursorStyle}`}>
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onWheel={handleWheel}
        onMouseLeave={() => { handleMouseUp(); setCrosshair(null); }}
        onDoubleClick={() => { setScaleY(1); setOffsetY(0); }}
        className="block"
      />
      {/* Toolbar - Added z-50 to fix visibility issues */}
      <div className="absolute top-4 left-1/2 transform -translate-x-1/2 flex gap-1 bg-[#1e222d] p-1 rounded border border-[#2a2e39] shadow-lg z-50">
          <button onClick={() => setDrawingMode('NONE')} className={`p-2 rounded ${drawingMode === 'NONE' ? 'bg-[#2962ff] text-white' : 'text-gray-400 hover:text-white'}`}>
            <span title="Cursor">Pointer</span>
          </button>
          <button onClick={() => setDrawingMode('LONG')} className={`p-2 rounded ${drawingMode === 'LONG' ? 'bg-[#2962ff] text-white' : 'text-gray-400 hover:text-white'}`}>
             <span title="Long Position">📈 Long</span>
          </button>
           <button onClick={() => setDrawingMode('SHORT')} className={`p-2 rounded ${drawingMode === 'SHORT' ? 'bg-[#2962ff] text-white' : 'text-gray-400 hover:text-white'}`}>
             <span title="Short Position">📉 Short</span>
          </button>
      </div>

      <div className="absolute top-2 left-2 flex flex-col gap-1 pointer-events-none z-40">
        <div className="text-xs text-gray-400 font-mono">
          {displayData.length > 0 ? `${displayData[0]?.original_symbol || ''} • ${timeframe}m • ${displayData.length} Bars` : 'No Data'}
        </div>
        <div className="text-[10px] text-gray-600">Double-click chart to reset zoom • Scroll to Zoom X</div>
      </div>
    </div>
  );
};

// =============================================================================
// 5. UI COMPONENTS
// =============================================================================

const NavButton = ({ active, onClick, icon, label }: { active: boolean, onClick: () => void, icon: string, label: string }) => (
    <button
        onClick={onClick}
        className={`w-10 h-10 mb-3 rounded-xl flex items-center justify-center text-lg transition-all ${
            active ? 'bg-[#2962ff] text-white shadow-lg scale-110' : 'text-gray-500 hover:text-white hover:bg-[#2a2e39]'
        }`}
        title={label}
    >
        {icon}
    </button>
);

const DashboardPanel = ({ state }: { state: AppState }) => (
    <div className="p-8 h-full overflow-y-auto">
        <h1 className="text-3xl font-bold text-white mb-6">QuantAI Studio Dashboard</h1>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
                { label: 'Active Dataset', value: state.datasets.find(d => d.id === state.activeDatasetId)?.name || 'None' },
                { label: 'Total Strategies', value: state.strategies.length },
                { label: 'Generated Setups', value: state.setups.length },
                { label: 'ML Models', value: state.models.length }
            ].map((stat, i) => (
                <div key={i} className="bg-[#1e222d] p-6 rounded-lg border border-[#2a2e39]">
                    <h3 className="text-gray-400 text-sm font-medium">{stat.label}</h3>
                    <p className="text-2xl font-bold text-white mt-2 truncate">{stat.value}</p>
                </div>
            ))}
        </div>
    </div>
);

const SetupsPanel = ({ state, setState, onRunAnalysis }: { state: AppState, setState: any, onRunAnalysis: (id: string) => void }) => (
    <div className="p-6 h-full overflow-y-auto">
        <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-white">Strategy Setups</h2>
            <button
                className="bg-[#2962ff] text-white px-4 py-2 rounded text-sm hover:bg-blue-600"
                onClick={() => {
                    const name = prompt("Strategy Name:");
                    if (name) {
                        setState((prev: AppState) => ({
                            ...prev,
                            strategies: [...prev.strategies, { id: `strat-${Date.now()}`, name, description: 'Manual Strategy', created: new Date() }]
                        }));
                    }
                }}
            >
                + New Strategy
            </button>
        </div>
        <div className="grid gap-4">
            {state.strategies.map(strat => (
                <div key={strat.id} className="bg-[#1e222d] p-4 rounded border border-[#2a2e39] flex justify-between items-center">
                    <div>
                        <h3 className="font-bold text-white">{strat.name}</h3>
                        <p className="text-sm text-gray-400">{strat.description}</p>
                    </div>
                    <div className="flex gap-2">
                        <button
                            onClick={() => setState((prev: AppState) => ({ ...prev, activeStrategyId: strat.id }))}
                            className={`px-3 py-1 rounded text-xs border ${state.activeStrategyId === strat.id ? 'border-[#2962ff] text-[#2962ff]' : 'border-gray-600 text-gray-400'}`}
                        >
                            {state.activeStrategyId === strat.id ? 'Active' : 'Select'}
                        </button>
                        <button
                            onClick={() => onRunAnalysis(strat.id)}
                            className="bg-[#2a2e39] hover:bg-[#363a45] text-white px-3 py-1 rounded text-xs"
                        >
                            Run Analysis
                        </button>
                    </div>
                </div>
            ))}
        </div>
    </div>
);

const TradesPanel = ({ state, setState }: { state: AppState, setState: any }) => {
    const downloadJSON = (tradeSet: TradeSet) => {
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(tradeSet, null, 2));
        const downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", `${tradeSet.name.replace(/\s+/g, '_')}_data.json`);
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    };

    return (
        <div className="p-6 h-full overflow-y-auto">
            <h2 className="text-2xl font-bold text-white mb-6">Manual Trade Lab</h2>
            <div className="grid gap-4">
                {state.tradeSets.length === 0 && <p className="text-gray-500">No manual trade sets created. Use the Chart to place trades.</p>}
                {state.tradeSets.map(ts => (
                    <div key={ts.id} className="bg-[#1e222d] p-4 rounded border border-[#2a2e39] flex flex-col gap-3">
                        <div className="flex justify-between items-center">
                            <div>
                                <h3 className="font-bold text-white">{ts.name}</h3>
                                <p className="text-xs text-gray-400">{ts.trades.length} trades • {ts.datasetId}</p>
                            </div>
                            <button
                                onClick={() => setState((prev: AppState) => ({ ...prev, activeTradeSetId: ts.id }))}
                                className={`px-3 py-1 rounded text-xs ${state.activeTradeSetId === ts.id ? 'bg-green-700 text-white' : 'bg-[#2a2e39] border border-gray-600 text-gray-300'}`}
                            >
                                {state.activeTradeSetId === ts.id ? 'Active' : 'Load'}
                            </button>
                        </div>
                        <div className="flex gap-2 justify-end">
                            <button 
                                onClick={() => downloadJSON(ts)}
                                className="text-xs bg-[#2a2e39] hover:bg-[#363a45] text-white px-3 py-2 rounded border border-gray-600"
                            >
                                ⬇ Export JSON
                            </button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

const DataPanel = ({ state, setState, backend }: { state: AppState, setState: any, backend: RealBackend }) => (
    <div className="p-6 h-full overflow-y-auto">
        <h2 className="text-2xl font-bold text-white mb-6">Data Management</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div className="bg-[#1e222d] p-4 rounded border border-[#2a2e39]">
                <h3 className="font-bold text-white mb-4">Generate Data</h3>
                <button
                    onClick={() => backend.generatePriceData("MOCK", 1000, 2.0)}
                    className="w-full bg-[#2962ff] text-white px-4 py-2 rounded text-sm hover:bg-blue-600"
                >
                    Generate Random (1000 bars)
                </button>
            </div>
            <div className="bg-[#1e222d] p-4 rounded border border-[#2a2e39]">
                <h3 className="font-bold text-white mb-4">Fetch External</h3>
                <div className="flex gap-2">
                    <button onClick={() => backend.fetchMarketData("BTC")} className="flex-1 bg-[#2a2e39] border border-gray-600 text-white px-3 py-2 rounded text-sm hover:bg-[#363a45]">BTC</button>
                    <button onClick={() => backend.fetchMarketData("ETH")} className="flex-1 bg-[#2a2e39] border border-gray-600 text-white px-3 py-2 rounded text-sm hover:bg-[#363a45]">ETH</button>
                </div>
            </div>
        </div>
        <h3 className="text-xl font-bold text-white mb-4">Datasets</h3>
        <div className="space-y-2">
            {state.datasets.map(d => (
                <div key={d.id} className="bg-[#1e222d] p-3 rounded border border-[#2a2e39] flex justify-between items-center">
                    <span className="text-sm text-gray-300">{d.name}</span>
                    <button
                        onClick={() => setState((prev: AppState) => ({ ...prev, activeDatasetId: d.id }))}
                        className={`px-2 py-1 rounded text-xs ${state.activeDatasetId === d.id ? 'bg-[#2962ff] text-white' : 'text-gray-400'}`}
                    >
                        {state.activeDatasetId === d.id ? 'Active' : 'Load'}
                    </button>
                </div>
            ))}
        </div>
    </div>
);

const ModelsPanel = ({ state }: { state: AppState, setState: any }) => (
    <div className="p-6 h-full overflow-y-auto">
        <h2 className="text-2xl font-bold text-white mb-6">Models</h2>
        <div className="grid gap-4">
            {state.models.length === 0 && <p className="text-gray-500">No models found.</p>}
            {state.models.map(m => (
                <div key={m.id} className="bg-[#1e222d] p-4 rounded border border-[#2a2e39] flex justify-between items-center">
                    <div>
                        <h3 className="font-bold text-white">{m.name}</h3>
                        <p className="text-xs text-gray-400">{m.type} • {m.status}</p>
                    </div>
                    <div className="text-right">
                        <div className="text-2xl font-bold text-[#26a69a]">{(m.accuracy * 100).toFixed(1)}%</div>
                        <div className="text-xs text-gray-500">Accuracy</div>
                    </div>
                </div>
            ))}
        </div>
    </div>
);

const ChatAgent = ({ backend }: { backend: RealBackend }) => {
    const [messages, setMessages] = useState<{ role: 'user' | 'model'; text: string }[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const chatRef = useRef<any>(null);
    const msgsEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const initChat = async () => {
             if (!process.env.API_KEY) return;
             
             const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
             
             const tools: any[] = [{
                functionDeclarations: [
                    {
                        name: "fetchMarketData",
                        description: "Fetch real market data for a symbol (BTC, ETH).",
                        parameters: {
                            type: Type.OBJECT,
                            properties: { symbol: { type: Type.STRING } },
                            required: ["symbol"]
                        }
                    },
                    {
                        name: "generatePriceData",
                        description: "Generate mock price data with specific tick size.",
                        parameters: {
                            type: Type.OBJECT,
                            properties: { 
                                symbol: { type: Type.STRING },
                                bars: { type: Type.NUMBER },
                                volatility: { type: Type.NUMBER }
                            },
                            required: ["symbol", "bars"]
                        }
                    },
                    {
                         name: "trainModel",
                         description: "Train a new model on a dataset.",
                         parameters: {
                             type: Type.OBJECT,
                             properties: {
                                 datasetId: { type: Type.STRING },
                                 modelType: { type: Type.STRING },
                                 epochs: { type: Type.NUMBER }
                             },
                             required: ["datasetId", "modelType"]
                         }
                    },
                    {
                        name: "execute_data_script",
                        description: "Execute a python data processing script. Use this for complex data prep.",
                        parameters: {
                            type: Type.OBJECT,
                            properties: {
                                scriptName: { type: Type.STRING, description: "Name of the script to run" },
                                args: { 
                                    type: Type.OBJECT, 
                                    description: "Dictionary of arguments like input_path, output_path, etc.",
                                    properties: {
                                        input_path: { type: Type.STRING },
                                        output_path: { type: Type.STRING },
                                        config_file: { type: Type.STRING },
                                        params: { type: Type.STRING }
                                    }
                                }
                            },
                            required: ["scriptName"]
                        }
                    },
                    {
                        name: "write_and_execute_code",
                        description: "Write and execute custom python code.",
                        parameters: {
                            type: Type.OBJECT,
                            properties: {
                                code: { type: Type.STRING, description: "The complete python code content." },
                                filename: { type: Type.STRING, description: "Filename to save as (e.g. 'custom_strategy.py')" }
                            },
                            required: ["code", "filename"]
                        }
                    }
                ]
             }];

             chatRef.current = ai.chats.create({
                 model: 'gemini-3-pro-preview',
                 config: {
                     systemInstruction: "You are a QuantAI assistant. You can generate data, train models, and execute arbitrary python scripts for data preparation. You can also write your own scripts and run them.",
                     thinkingConfig: { thinkingBudget: 32768 },
                     tools: tools
                 }
             });
        };
        initChat();
    }, [backend]); // Re-init if backend instance changes

    const handleSend = async () => {
        if (!input || !chatRef.current) return;
        const msg = input;
        setInput('');
        setMessages(prev => [...prev, { role: 'user', text: msg }]);
        setLoading(true);
        
        try {
            let response = await chatRef.current.sendMessage({ message: msg });
            
            // Handle Tool Calls
            while (response.functionCalls && response.functionCalls.length > 0) {
                 const calls = response.functionCalls;
                 const responses = [];
                 
                 for (const call of calls) {
                     let result = "Error";
                     try {
                         if (call.name === 'fetchMarketData') {
                             result = await backend.fetchMarketData(call.args.symbol);
                         } else if (call.name === 'generatePriceData') {
                             result = await backend.generatePriceData(call.args.symbol, call.args.bars || 1000, call.args.volatility || 2);
                         } else if (call.name === 'trainModel') {
                             result = await backend.trainModel(call.args.datasetId, call.args.modelType, call.args.epochs || 10);
                         } else if (call.name === 'execute_data_script') {
                             result = await backend.executeDataScript(call.args.scriptName, call.args.args || {});
                         } else if (call.name === 'write_and_execute_code') {
                             result = await backend.writeAndExecuteCode(call.args.code, call.args.filename);
                         }
                     } catch (e: any) {
                         result = `Error: ${e.message}`;
                     }
                     responses.push({
                         functionResponse: {
                             name: call.name,
                             response: { result: result },
                             id: call.id // Pass ID if available
                         }
                     });
                 }
                 
                 response = await chatRef.current.sendMessage(responses);
            }
            
            setMessages(prev => [...prev, { role: 'model', text: response.text }]);
        } catch (e: any) {
            setMessages(prev => [...prev, { role: 'model', text: `Error: ${e.message}` }]);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => msgsEndRef.current?.scrollIntoView(), [messages]);

    return (
        <div className="flex flex-col h-full bg-[#1e222d] border-l border-[#2a2e39]">
            <div className="p-4 bg-[#2a2e39] font-bold text-white border-b border-[#363a45]">Gemini Assistant</div>
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((m, i) => (
                    <div key={i} className={`p-3 rounded text-sm max-w-[90%] ${m.role === 'user' ? 'bg-[#2962ff] text-white self-end ml-auto' : 'bg-[#2a2e39] text-gray-200'}`}>
                        {m.text}
                    </div>
                ))}
                {loading && <div className="text-xs text-gray-500 animate-pulse">Thinking...</div>}
                <div ref={msgsEndRef} />
            </div>
            <div className="p-3 border-t border-[#2a2e39] flex gap-2">
                <input 
                    className="flex-1 bg-[#131722] border border-[#363a45] rounded px-2 py-1 text-white text-sm focus:outline-none focus:border-[#2962ff]"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleSend()}
                    placeholder="Type a command..."
                />
                <button onClick={handleSend} disabled={loading} className="text-[#2962ff] hover:text-white px-2">➤</button>
            </div>
        </div>
    );
};

const App = () => {
  // ... (State same as before)
  const [activeTab, setActiveTab] = useState<'home' | 'chart' | 'data' | 'models' | 'setups' | 'trades'>('home');
  const [state, setState] = useState<AppState>({
    datasets: [
      {
        id: 'ds-seed',
        name: 'Seed Data (MESM5)',
        source: 'MOCK',
        created: new Date(),
        data: SEED_DATA
      }
    ],
    models: [],
    strategies: [
       { id: 'strat-sample-1', name: 'RSI Divergence', description: 'Reversal setup based on RSI div', created: new Date() }
    ],
    tradeSets: [],
    activeDatasetId: 'ds-seed',
    activeModelId: null,
    activeStrategyId: 'strat-sample-1',
    activeTradeSetId: null,
    setups: [],
    chartTimeframe: 1
  });

  const activeData = useMemo(() => 
    state.datasets.find(d => d.id === state.activeDatasetId)?.data || [], 
  [state.activeDatasetId, state.datasets]);

  // Use a ref to keep the backend stable, but allowing it to access latest state
  const stateRef = useRef(state);
  useEffect(() => { stateRef.current = state; }, [state]);

  const backend = useMemo(() => new RealBackend(stateRef, setState), []);

  const handleRunAnalysis = (strategyId: string) => {
      if (!state.activeDatasetId) return;
      backend.runSetupAnalysis(strategyId, state.activeDatasetId, 2.0);
      setActiveTab('chart');
  }

  return (
    <div className="flex w-full h-screen bg-[#131722] text-[#d1d4dc] font-sans overflow-hidden">
      {/* ... Sidebar ... */}
      <div className="w-16 flex flex-col items-center py-4 bg-[#1e222d] border-r border-[#2a2e39] z-10">
        <div className="mb-6 font-bold text-[#2962ff] text-xl">QAI</div>
        <NavButton active={activeTab === 'home'} onClick={() => setActiveTab('home')} icon="🏠" label="Home" />
        <NavButton active={activeTab === 'chart'} onClick={() => setActiveTab('chart')} icon="📈" label="Chart" />
        <NavButton active={activeTab === 'trades'} onClick={() => setActiveTab('trades')} icon="🧪" label="Lab" />
        <NavButton active={activeTab === 'setups'} onClick={() => setActiveTab('setups')} icon="🎯" label="Setups" />
        <NavButton active={activeTab === 'data'} onClick={() => setActiveTab('data')} icon="💾" label="Data" />
        <NavButton active={activeTab === 'models'} onClick={() => setActiveTab('models')} icon="🧠" label="Models" />
      </div>

      <div className="flex-1 flex flex-col min-w-0">
        {/* ... Header ... */}
        <div className="h-12 bg-[#1e222d] border-b border-[#2a2e39] flex items-center px-4 justify-between">
          <div className="flex items-center gap-4">
             <div className="font-semibold text-white">
                {state.activeDatasetId ? state.datasets.find(d => d.id === state.activeDatasetId)?.name : "No Data Selected"}
             </div>
             {state.activeStrategyId && (
                 <div className="text-xs bg-[#2962ff] text-white px-2 py-1 rounded">
                     Active Strat: {state.strategies.find(s => s.id === state.activeStrategyId)?.name}
                 </div>
             )}
             {state.activeTradeSetId && (
                 <div className="text-xs bg-green-700 text-white px-2 py-1 rounded">
                     Manual Set: {state.tradeSets.find(s => s.id === state.activeTradeSetId)?.name}
                 </div>
             )}
          </div>
          
          {activeTab === 'chart' && (
              <div className="flex gap-1 bg-[#131722] rounded p-1 border border-[#2a2e39]">
                  {[1, 5, 15, 60].map(tf => (
                      <button
                        key={tf}
                        onClick={() => setState(prev => ({...prev, chartTimeframe: tf}))}
                        className={`px-3 py-1 text-xs rounded transition ${state.chartTimeframe === tf ? 'bg-[#2962ff] text-white' : 'text-gray-400 hover:text-white'}`}
                      >
                        {tf === 60 ? '1H' : `${tf}m`}
                      </button>
                  ))}
              </div>
          )}
        </div>

        <div className="flex-1 relative overflow-hidden">
          {activeTab === 'home' && <DashboardPanel state={state} />}
          {activeTab === 'chart' && (
            <div className="w-full h-full relative">
              <CanvasChart 
                data={activeData} 
                setups={state.setups} 
                manualSets={state.tradeSets}
                activeStrategyId={state.activeStrategyId} 
                activeTradeSetId={state.activeTradeSetId}
                timeframe={state.chartTimeframe}
                onAddTrade={(trade) => {
                    if (state.activeDatasetId) {
                        backend.addManualTrade(trade, state.activeDatasetId);
                    }
                }}
                onUpdateTrade={(trade) => backend.updateManualTrade(trade)}
              />
              <div className="absolute top-4 right-4 bg-[#1e222d] p-3 rounded shadow-lg border border-[#2a2e39] w-64 opacity-90 z-40">
                 <h4 className="font-bold text-xs text-gray-300 mb-2">Display Filter</h4>
                 <select 
                   className="w-full bg-[#2a2e39] text-xs text-white p-2 rounded border border-[#363a45]"
                   value={state.activeStrategyId || ''}
                   onChange={(e) => setState(prev => ({ ...prev, activeStrategyId: e.target.value }))}
                 >
                    <option value="">Show All Setups</option>
                    {state.strategies.map(s => (
                        <option key={s.id} value={s.id}>{s.name}</option>
                    ))}
                 </select>
              </div>
            </div>
          )}
          {activeTab === 'setups' && <SetupsPanel state={state} setState={setState} onRunAnalysis={handleRunAnalysis} />}
          {activeTab === 'trades' && <TradesPanel state={state} setState={setState} />}
          {activeTab === 'data' && <DataPanel state={state} setState={setState} backend={backend} />}
          {activeTab === 'models' && <ModelsPanel state={state} setState={setState} />}
        </div>
      </div>
      
      <div className="w-[350px] h-full shadow-xl z-20 flex-shrink-0">
        <ChatAgent backend={backend} />
      </div>
    </div>
  );
};

// Mount Application
const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}