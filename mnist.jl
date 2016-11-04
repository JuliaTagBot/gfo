# check out the following for active learning:
# https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-867-machine-learning-fall-2006/lecture-notes/lec6.pdf
# says we need to pick the largest eigenvector of A=(X'X)^-1 where X has row vectors for all previous examples.
# note the closed form solution for linreg is w=AX'y
# all this assumes X'X is invertible.  In our case X is fat and it is probably not.

# https://en.wikipedia.org/wiki/Active_learning_(machine_learning)
# says svm active learning picks points closest to the hyperplane or furthest or alternate.
# closest suggests picking vecs orthogonal to wpred - in 10K dimensions any random vector will do.
# it may not be possible to beat random vector.

using Knet
isdefined(:MNIST) || include(Knet.dir("examples/mnist.jl"))
using MNIST: minibatch,xtst,ytst,xtrn,ytrn
(x0,y0) = minibatch(xtst, ytst, 10000; atype=KnetArray{Float32})[1]
loss(w) = -sum(y0 .* logp(w*x0,1))/size(y0,2)
gloss = grad(loss)

# This version gets stuck around loss=0.5.
# TODO: update gpred and hpred based on f3, the move in gpred direction.
# replacing the ad-hoc direction flip.

function gfo(; batch=100, lr=.01f0, epochs=100, atype=KnetArray{Float32}, vstd=1e-5, lr_g=1.0, lr_h=1.0, hpred=1.0)
    Knet.rng(true)
    dtrn = minibatch(xtrn,ytrn,batch;atype=atype)
    dtst = minibatch(xtst,ytst,length(ytst);atype=atype)
    f(w,x,y) = -sum(y .* logp(w*x,1))/size(y,2); g = grad(f)
    w = convert(atype, zeros(10,784)); vrnd = similar(w); gpred = copy(w); lr0=lr
    println((0, mean(f(w,x,y) for (x,y) in dtst)))
    for epoch=1:epochs
        for (x,y) in dtrn
            f0 = f(w,x,y)

            # adjust gpred using random step
            v = randn!(vrnd,0,vstd)
            vv = sumabs2(v)
            w1 = w+v
            f1 = f(w1,x,y)
            dfgold = f1-f0
            dfpred = dot(gpred,v) # + 0.5 * hpred * vv: hpred only accurate in gpred direction, v is small etc.
            delta = (dfpred - dfgold)
            gpred -= (lr_g * delta / (2*vv)) * v
            # hpred -= (lr_h * delta / vv)

            # adjust hpred using newton step
            v = (-1/hpred)*gpred
            vv = sumabs2(v)
            w2 = w+v
            f2 = f(w2,x,y)
            dfgold = f2-f0
            dfpred = dot(gpred,v) + 0.5 * hpred * vv
            delta = (dfpred - dfgold)
            gpred -= (lr_g * delta / (2*vv)) * v
            hpred -= (lr_h * delta / vv)

            if f2 < f0
                axpy!(-1/hpred, gpred, w)
            else
                gpred *= -1
                lr *= 0.99f0
                lr < (1e-6) && (lr=lr0; fill!(gpred,0))
            end
        end
        println((epoch,
                 :lr,lr,
                 :loss, mean(f(w,x,y) for (x,y) in dtst),
                 :vcos, mean(vcos(gpred,g(w,x,y)) for (x,y) in dtst),
                 :ggold, mean(vecnorm(g(w,x,y)) for (x,y) in dtst),
                 :gpred, vecnorm(gpred), :hpred, hpred
                 ))
    end
end

function sgd(; batch=100, lr=batch/1000, epochs=10, atype=KnetArray{Float32})
    dtrn = minibatch(xtrn,ytrn,batch;atype=atype)
    dtst = minibatch(xtst,ytst,length(ytst);atype=atype)
    f(w,x,y) = -sum(y .* logp(w*x,1))/size(y,2)
    g = grad(f)
    w = convert(atype, zeros(10,784))
    println((0, mean([f(w,x,y) for (x,y) in dtst])))
    for epoch=1:epochs
        for (x,y) in dtrn
            dw = g(w,x,y)
            axpy!(-lr,dw,w)
        end
        losses = [ f(w,x,y) for (x,y) in dtst ]
        println((epoch, mean([f(w,x,y) for (x,y) in dtst])))
    end
end

# measure gradient variance
function gradvar(n=100)
    f(w,x,y) = -sum(y .* logp(w*x,1))/size(y,2)
    g = grad(f)
    data = minibatch(xtrn,ytrn,n; atype=Array{Float64})
    grads = Any[]
    w = oftype(data[1][1], zeros(10,784))
    for (x,y) in data
        push!(grads, g(w,x,y))
    end
    g0 = mean(grads)
    cosine = Any[]
    sqdiff = Any[]
    for g in grads
        push!(cosine, vcos(g,g0))
        push!(sqdiff, sumabs2(g-g0))
    end
    (:cos, mean(cosine), :std, sqrt(mean(sqdiff)), :nrm, mean(map(vecnorm,grads)))
end

vcos(a,b)=dot(a,b)/(vecnorm(a)*vecnorm(b))

function flinreg(w,x,y;l1=0,l2=0,l3=0)
    lss = zero(eltype(w))
    if l1 != 0; (lss += l1*sumabs(w)); end
    if l2 != 0; (lss += l2*sumabs2(w)); end
    if l3 != 0 # prevent l1 reg from changing the sign
        lss += sum(min(0.5*abs2(w),l3*abs(w)))
    end
    lss += abs2(y - sum(w.*x))
    return lss
end

glinreg = grad(flinreg)

# with no l1/l2: 10.18 10000 (:cos,0.74111754f0)
# best l1=1e-14 11.24 10000 (:cos,0.7515351f0)
# no meaningful improvement with l2.

function linreg(epochs=10000;l1=1e-14,l2=0,l3=0,lr=0.5)
    Knet.rng(true)
    w0 = oftype(x0, zeros(10,784))
    f0 = loss(w0)
    g0 = gloss(w0)
    gpred = oftype(x0, zeros(10,784))
    dw = similar(w0)
    progress(epoch)=(:cos,dot(gpred,g0)/(vecnorm(gpred)*vecnorm(g0)))
    report()
    for epoch=1:epochs
        randn!(dw, 0, 1e-6)
        w1 = w0 + dw
        f1 = loss(w1)
        dgold = f1 - f0
        ggrad = glinreg(gpred, dw, dgold; l1=l1, l2=l2, l3=l3)
        # axpy!(-lr, ggrad, gpred)
        # See the changelog 2016-11-03 for the sumabs2(dw) factor.
        axpy!(-lr/sumabs2(dw), ggrad, gpred)
        report(progress, epoch)
    end
    report(progress, epochs, final=true)
    return gpred
end

function flogreg(w,x,y,l1,l2)
    lss = zero(eltype(w))
    if l1 != 0; (lss += l1*sumabs(w)); end
    if l2 != 0; (lss += l2*sumabs2(w)); end
    prob = sigm(sum(w.*x)) #sigm(dot(w,x))
    if sign(y) < 0; (prob = 1-prob); end
    lss -= log(prob)
    return lss
end

glogreg = grad(flogreg)

# l1=3e-10 works best in epochs=1000,lr=1 -- 1.09 1000 (:cos,0.26973265f0)
# compared to no l1/l2: 1.15 1000 (:cos,0.26972535f0)
# compared to perceptron: 1.07 1000 (:cos,0.21155934f0,:acc,0.5769524461878847)
# l1=3e-9 works best in epochs=10000,lr=1 -- 10.69 10000 (:cos,0.6744472f0)
# l2=1e-6 works best in epochs=10000,lr=1 -- 10.68 10000 (:cos,0.66643745f0)
# compared to no l1/l2: 9.97 10000 (:cos,0.66643345f0)
# compared to perceptron: 20.90 10000 (:cos,0.5908562f0,:acc,0.7445132059813894)
function logreg(epochs=10000;l1=3e-9,l2=0,lr=1.0)
    Knet.rng(true)
    w0 = oftype(x0, zeros(10,784))
    f0 = loss(w0)
    g0 = gloss(w0)
    gpred = oftype(x0, zeros(10,784))
    dw = similar(w0)
    progress(epoch)=(:cos,dot(gpred,g0)/(vecnorm(gpred)*vecnorm(g0)))
    report()
    for epoch=1:epochs
        randn!(dw, 0, 1e-6)
        w1 = w0 + dw
        f1 = loss(w1)
        dgold = f1 - f0
        ggrad = glogreg(gpred, dw, dgold, l1, l2)
        axpy!(-lr, ggrad, gpred)
        report(progress, epoch)
    end
    report(progress, epochs, final=true)
    return gpred
end

function perceptron(epochs=100000)
    w0 = oftype(x0, zeros(10,784))
    f0 = loss(w0)
    g0 = gloss(w0)
    gpred = oftype(x0, randn(10,784)*1e-6)
    dw = similar(w0)
    acc = 0.5
    progress(epoch)=(:cos,dot(gpred,g0)/(vecnorm(gpred)*vecnorm(g0)),:acc,acc)
    report()
    for epoch=1:epochs
        randn!(dw, 0, 1e-6)
        w1 = w0 + dw
        f1 = loss(w1)
        dgold = f1 - f0
        dpred = dot(gpred,dw)
        acc = 0.999 * acc
        if dpred * dgold < 0
            gpred += sign(dgold) * dw
        else
            acc += 0.001
        end
        report(progress, epoch)
    end
    report(progress, epochs, final=true)
    return gpred
end

function spsa(epochs=100000)
    w0 = oftype(x0, zeros(10,784))
    f0 = loss(w0)
    g0 = gloss(w0)
    gpred = oftype(x0, randn(10,784)*1e-6)
    dw = similar(w0)
    acc = 0.5
    progress(epoch)=(:cos,dot(gpred,g0)/(vecnorm(gpred)*vecnorm(g0)),:acc,acc)
    report()
    for epoch=1:epochs
        dw = oftype(x0, rand(-1e6:2e6:1e6, size(w0)))
        w1 = w0 + dw
        f1 = loss(w1)
        dgold = f1 - f0
        dpred = dot(gpred,dw)
        acc = 0.999 * acc + 0.001 * (dpred * dgold >= 0)
        gpred += dgold ./ dw
        report(progress, epoch)
    end
    report(progress, epochs, final=true)
    return gpred
end


let time0=tnext=nnext=1
    global report
    report()=(time0=time();tnext=time0;nnext=1)
    function report(f,n; dt=10, dn=2, final=false)
        tnext == 1 && report()
        if final || n >= nnext || time() >= tnext
            t = time()
            @printf("%.2f %d %s\n", t-time0, n, f(n))
            t >= tnext && (tnext = t + dt)
            n >= nnext && (nnext *= dn)
        end
    end
end
    
macro cuda1(lib,fun,x...)
    if Libdl.find_library(["lib$lib"], []) == ""
        msg = "Cannot find lib$lib, please install it and rerun Pkg.build(\"Knet\")."
        :(error($msg))
    else
        f2 = ("$fun","lib$lib")
        fx = Expr(:ccall, f2, :UInt32, x...)
        err = "$lib.$fun error "
        quote
            local _r = $fx
            if _r != 0
                warn($err, _r)
                Base.show_backtrace(STDOUT, backtrace())
            end
        end
    end
end

using Knet: Cptr, rng, cublashandle

import Base: randn!
randn!(a::KnetArray{Float32},mean,stddev)=(@cuda1(curand,curandGenerateNormal,(Cptr,Ptr{Float32},Csize_t,Float32,Float32),rng(),a,length(a),mean,stddev); a)
randn!(a::KnetArray{Float64},mean,stddev)=(@cuda1(curand,curandGenerateNormalDouble,(Cptr,Ptr{Float64},Csize_t,Float64,Float64),rng(),a,length(a),mean,stddev); a)
randn!(a::Array,mean,stddev)=(randn!(a);stddev!=1 && scale!(a,stddev);mean!=0 && (a[:]+=mean);a)

import Base.LinAlg: dot
dot(x::KnetArray{Float32},y::KnetArray{Float32})=(c=Float32[0];@cuda1(cublas,cublasSdot_v2, (Cptr, Cint, Ptr{Float32}, Cint, Ptr{Float32}, Cint, Ptr{Float32}), cublashandle, length(x), x, 1, y, 1, c);c[1])
dot(x::KnetArray{Float64},y::KnetArray{Float64})=(c=Float64[0];@cuda1(cublas,cublasDdot_v2, (Cptr, Cint, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Ptr{Float64}), cublashandle, length(x), x, 1, y, 1, c);c[1])

import Knet: sigmback
sigmback(a::Number,b::Number)=sigmback(promote(a,b)...)
