"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[2942],{3905:function(e,t,n){n.d(t,{Zo:function(){return p},kt:function(){return m}});var i=n(7294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);t&&(i=i.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,i)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,i,r=function(e,t){if(null==e)return{};var n,i,r={},a=Object.keys(e);for(i=0;i<a.length;i++)n=a[i],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(i=0;i<a.length;i++)n=a[i],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var c=i.createContext({}),u=function(e){var t=i.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},p=function(e){var t=u(e.components);return i.createElement(c.Provider,{value:t},e.children)},l={inlineCode:"code",wrapper:function(e){var t=e.children;return i.createElement(i.Fragment,{},t)}},f=i.forwardRef((function(e,t){var n=e.components,r=e.mdxType,a=e.originalType,c=e.parentName,p=s(e,["components","mdxType","originalType","parentName"]),f=u(n),m=r,d=f["".concat(c,".").concat(m)]||f[m]||l[m]||a;return n?i.createElement(d,o(o({ref:t},p),{},{components:n})):i.createElement(d,o({ref:t},p))}));function m(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var a=n.length,o=new Array(a);o[0]=f;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s.mdxType="string"==typeof e?e:r,o[1]=s;for(var u=2;u<a;u++)o[u]=n[u];return i.createElement.apply(null,o)}return i.createElement.apply(null,n)}f.displayName="MDXCreateElement"},5352:function(e,t,n){n.r(t),n.d(t,{assets:function(){return p},contentTitle:function(){return c},default:function(){return m},frontMatter:function(){return s},metadata:function(){return u},toc:function(){return l}});var i=n(7462),r=n(3366),a=(n(7294),n(3905)),o=["components"],s={id:"jit",title:"Just In Time Optimization",slug:"/framework/jit"},c=void 0,u={unversionedId:"framework/jit/jit",id:"framework/jit/jit",title:"Just In Time Optimization",description:"Mass-Spring System with Just In Time Optimization",source:"@site/docs/framework/jit/jit.md",sourceDirName:"framework/jit",slug:"/framework/jit",permalink:"/docs/framework/jit",draft:!1,tags:[],version:"current",frontMatter:{id:"jit",title:"Just In Time Optimization",slug:"/framework/jit"},sidebar:"diffktSidebar",previous:{title:"Vector-Jacobian",permalink:"/docs/framework/vector_jacobian"},next:{title:"ShapeKt",permalink:"/docs/framework/shapekt"}},p={},l=[{value:"Jit Tips and Tricks",id:"jit-tips-and-tricks",level:2}],f={toc:l};function m(e){var t=e.components,n=(0,r.Z)(e,o);return(0,a.kt)("wrapper",(0,i.Z)({},f,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("div",{className:"admonition admonition-tip alert alert--success"},(0,a.kt)("div",{parentName:"div",className:"admonition-heading"},(0,a.kt)("h5",{parentName:"div"},(0,a.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,a.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"},(0,a.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"}))),"Open tutorial in Github")),(0,a.kt)("div",{parentName:"div",className:"admonition-content"},(0,a.kt)("p",{parentName:"div"},(0,a.kt)("a",{parentName:"p",href:"https://github.com/facebookresearch/diffkt/blob/main/tutorials/mass_spring_jit.ipynb"},"Mass-Spring System with Just In Time Optimization")))),(0,a.kt)("p",null,"The Just In Time (jit) optimization api produces a optimized version of ",(0,a.kt)("strong",{parentName:"p"},"DiffKt")," code. This is useful\nfor when you repeatedly call a function. On the first call to a jitted function, an optimized\nversion is created. On subsequent calls, the optimized version is called, which should result\nin a speed up of the program."),(0,a.kt)("h2",{id:"jit-tips-and-tricks"},"Jit Tips and Tricks"),(0,a.kt)("p",null,"There are lots of subtle things you need to get right to take full advantage of the jit:"),(0,a.kt)("p",null,"Make sure there is a good ",(0,a.kt)("inlineCode",{parentName:"p"},"equals()")," and ",(0,a.kt)("inlineCode",{parentName:"p"},"hashCode()")," function for the jitted\nfunction's input type. The jit cache needs that."),(0,a.kt)("p",null,"For the purposes of the jit, wrapping more of the input is better. For example, if you\nhave some inputs that are not active variables of differentiation inside the body of\nthe jitted function, it is still valuable to wrap them for the purposes of the jit so that\nyou will get a cache hit when the values change. That means you may want to use a\ndifferent (explicit) wrapInput lambda when taking the derivative."),(0,a.kt)("p",null,"Don't use mutable variables from an enclosing scope. If they are var variables\n(i.e. they don't change) that is OK, but if the value might change from call to call\nof the jitted function, they should be explicit inputs to the function."),(0,a.kt)("p",null,"Don't have side-effects in the jitted function; it should be a pure function.\nThat means no print statements, random number generation, or taking the time of day."))}m.isMDXComponent=!0}}]);