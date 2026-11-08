"use client";

import { useState } from "react";

// Matches the ListingResponse schema from FastAPI
interface Proof {
  category: string;
  description: string;
}

interface AnalysisResult {
  scam_score: number;
  risk_level: string;
  proofs: Proof[];
  action_required: string;
}

export default function Home() {
  const [url, setUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCheckListing = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/check-listing", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      if (!response.ok) {
        throw new Error("Failed to analyze listing. Check the URL and try again.");
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Determine meter color based on risk thresholds
  const getScoreColor = (score: number) => {
    if (score < 40) return "text-green-500";
    if (score < 70) return "text-yellow-500";
    return "text-red-500";
  };

  return (
    <main className="min-h-screen bg-gray-50 p-8 font-sans">
      <div className="max-w-3xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight text-gray-900">RentRakshak</h1>
          <p className="text-gray-500">Pre-Visit Rental Fraud Guard for NoBroker & 99acres</p>
        </div>

        {/* Input Form[cite: 10] */}
        <form onSubmit={handleCheckListing} className="bg-white p-6 rounded-xl shadow-sm border space-y-4">
          <div>
            <label htmlFor="url" className="block text-sm font-medium text-gray-700 mb-1">
              Property Listing URL
            </label>
            <input
              id="url"
              type="url"
              required
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Paste NoBroker or 99acres link here..."
              className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all text-black"
            />
          </div>
          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition-colors disabled:opacity-50"
          >
            {isLoading ? "Running Forensics (Takes ~5s)..." : "Analyze Risk"}
          </button>
        </form>

        {/* Error State */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg">
            {error}
          </div>
        )}

        {/* Results UI */}
        {result && (
          <div className="space-y-6">
            
            {/* Score Meter[cite: 10] */}
            <div className="bg-white p-8 rounded-xl shadow-sm border text-center space-y-2">
              <h2 className="text-lg font-medium text-gray-500 uppercase tracking-wide">Scam Score</h2>
              <div className={`text-7xl font-black ${getScoreColor(result.scam_score)}`}>
                {result.scam_score}
                <span className="text-3xl text-gray-400 font-medium">/100</span>
              </div>
              <div className="text-xl font-bold text-gray-800 mt-2">{result.risk_level}</div>
              <p className="text-gray-600 mt-4 bg-gray-50 p-4 rounded-lg inline-block w-full">
                <strong>Action Required:</strong> {result.action_required}
              </p>
            </div>

            {/* 3 Proof Cards[cite: 10] */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {result.proofs.map((proof, idx) => (
                <div key={idx} className="bg-white p-5 rounded-xl shadow-sm border border-gray-100 flex flex-col h-full">
                  <h3 className="text-sm font-bold text-blue-600 uppercase tracking-wider mb-3">
                    {proof.category} Check
                  </h3>
                  <p className="text-gray-700 text-sm leading-relaxed flex-grow">
                    {proof.description}
                  </p>
                </div>
              ))}
            </div>

          </div>
        )}
      </div>
    </main>
  );
}