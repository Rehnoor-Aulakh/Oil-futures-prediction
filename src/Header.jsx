import React from "react";

export default function Header() {
  return (
    <header className="text-center flex flex-col items-center">
      <img src="logo.png" alt="" className="w-20" />
      <h1 className="mt-6 text-center text-3xl text-cyan-400 font-bold">
        Energy Market Analysis
      </h1>
    </header>
  );
}
