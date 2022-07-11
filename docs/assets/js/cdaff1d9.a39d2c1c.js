"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[9280],{3905:function(e,t,n){n.d(t,{Zo:function(){return p},kt:function(){return d}});var r=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function u(e,t){if(null==e)return{};var n,r,i=function(e,t){if(null==e)return{};var n,r,i={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var c=r.createContext({}),l=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},p=function(e){var t=l(e.components);return r.createElement(c.Provider,{value:t},e.children)},s={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},f=r.forwardRef((function(e,t){var n=e.components,i=e.mdxType,a=e.originalType,c=e.parentName,p=u(e,["components","mdxType","originalType","parentName"]),f=l(n),d=i,v=f["".concat(c,".").concat(d)]||f[d]||s[d]||a;return n?r.createElement(v,o(o({ref:t},p),{},{components:n})):r.createElement(v,o({ref:t},p))}));function d(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var a=n.length,o=new Array(a);o[0]=f;var u={};for(var c in t)hasOwnProperty.call(t,c)&&(u[c]=t[c]);u.originalType=e,u.mdxType="string"==typeof e?e:i,o[1]=u;for(var l=2;l<a;l++)o[l]=n[l];return r.createElement.apply(null,o)}return r.createElement.apply(null,n)}f.displayName="MDXCreateElement"},3544:function(e,t,n){n.r(t),n.d(t,{assets:function(){return p},contentTitle:function(){return c},default:function(){return d},frontMatter:function(){return u},metadata:function(){return l},toc:function(){return s}});var r=n(7462),i=n(3366),a=(n(7294),n(3905)),o=["components"],u={id:"quick_start",title:"Quick Start",slug:"/overview/quick_start"},c=void 0,l={unversionedId:"overview/quick_start/quick_start",id:"overview/quick_start/quick_start",title:"Quick Start",description:"On a Mac",source:"@site/docs/overview/quick_start/quick_start.md",sourceDirName:"overview/quick_start",slug:"/overview/quick_start",permalink:"/docs/overview/quick_start",draft:!1,tags:[],version:"current",frontMatter:{id:"quick_start",title:"Quick Start",slug:"/overview/quick_start"},sidebar:"diffktSidebar",previous:{title:"Automatic Differentiation",permalink:"/docs/overview/automatic_differentiation"},next:{title:"Installation on Mac",permalink:"/docs/overview/installation_mac"}},p={},s=[{value:"On a Mac",id:"on-a-mac",level:2},{value:"Maven",id:"maven",level:3},{value:"Gradle/JVM",id:"gradlejvm",level:3},{value:"On Ubuntu",id:"on-ubuntu",level:2},{value:"With a GPU",id:"with-a-gpu",level:2}],f={toc:s};function d(e){var t=e.components,n=(0,i.Z)(e,o);return(0,a.kt)("wrapper",(0,r.Z)({},f,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h2",{id:"on-a-mac"},"On a Mac"),(0,a.kt)("h3",{id:"maven"},"Maven"),(0,a.kt)("p",null,"A precompiled ",(0,a.kt)("strong",{parentName:"p"},"DiffKt")," jar for a Mac is available in Maven. It has jni libraries included,\nso it only works on a Mac. This jar does not include the jni for using a GPU, so do not enable the GPU\nin the code."),(0,a.kt)("p",null,"The current version is ",(0,a.kt)("inlineCode",{parentName:"p"},"0.0.1-DEV2")),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"https://search.maven.org/artifact/com.facebook.diffkt/diffkt/0.0.1-DEV2/jar"},"Maven Description")),(0,a.kt)("h3",{id:"gradlejvm"},"Gradle/JVM"),(0,a.kt)("p",null,"To use DiffKt, use the following dependency to your ",(0,a.kt)("inlineCode",{parentName:"p"},"build.gradle.kts")," file with the ",(0,a.kt)("inlineCode",{parentName:"p"},"x.y.z")," version number."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre"},'dependencies {\n    implementation("com.facebook.diffkt:x.y.z")\n}\n')),(0,a.kt)("h2",{id:"on-ubuntu"},"On Ubuntu"),(0,a.kt)("p",null,"Currently, you need to checkout the repo and build it."),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"installation_ubuntu"},"Ubuntu Installation")),(0,a.kt)("h2",{id:"with-a-gpu"},"With a GPU"),(0,a.kt)("p",null,"Currently, you need to checkout the repo and build it."))}d.isMDXComponent=!0}}]);