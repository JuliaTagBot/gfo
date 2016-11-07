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
vcos(a,b)=dot(a,b)/(vecnorm(a)*vecnorm(b))

# This version gets stuck around loss=0.4 in 100 epochs with batch=100.

function gfo(w=nothing; batch=100, epochs=100, atype=KnetArray{Float64}, vstd=1e-5, lr_g=1.0, lr_h=1.0, hpred=0.01, pr=0, lr=0.05, decay=0.8)
    Knet.rng(true)
    dtrn = minibatch(xtrn,ytrn,batch;atype=atype)
    dtst = minibatch(xtst,ytst,length(ytst);atype=atype)
    f(w,x,y) = -sum(y .* logp(w*x,1))/size(y,2); gradf = grad(f)
    w==nothing && (w = convert(atype, zeros(10,784)))
    vrnd = zeros(w); gpred = zeros(w); hpred0=hpred; lr0=lr; loss0=loss1=Inf
    report(epoch)=println(map((epoch,
                               :lr, lr,
                               :loss, mean(f(w,x,y) for (x,y) in dtst),
                               :vcos, mean(vcos(gpred,gradf(w,x,y)) for (x,y) in dtst),
                               :ggold, mean(vecnorm(gradf(w,x,y)) for (x,y) in dtst),
                               :gpred, vecnorm(gpred), :hpred, hpred
                               )) do x; isa(x,AbstractFloat) ? parse(@sprintf("%g",x)) : x; end)
    report(0)
    approx(a,b)=isapprox(a,b;rtol=1e-3,atol=1e-12)
    for epoch=1:epochs
        for (x,y) in dtrn
            f0 = f(w,x,y)
            g0 = gradf(w,x,y)

            # adjust direction using small random step
            v = randn!(vrnd,0,vstd)
            vv = sumabs2(v)
            w1 = w+v
            f1 = f(w1,x,y)
            dfgold = f1-f0
            dfpred = dot(gpred,v)
            delta = (dfpred - dfgold)
            vcos1 = vcos(g0,gpred)
            gpred -= (lr_g * delta / vv) * v
            vcos2 = vcos(g0,gpred)
            approx(dfgold, dot(gpred,v)) || error("dfgold1=$dfgold dfpred1=$(dot(gpred,v))")

            # adjust size using small g step
            v = (-vecnorm(v)/vecnorm(gpred))*gpred
            approx(vv, sumabs2(v)) || error("vv=$vv sumabs2(v)=$(sumabs2(v))")
            w2 = w+v
            f2 = f(w2,x,y)
            # f2 < f0 || warn("f0=$f0 f2=$f2") # f may not decrease when w changes
            dfgold = f2-f0
            dfpred = dot(gpred,v)
            delta = (dfpred - dfgold)
            size1 = vecnorm(gpred)
            size0 = abs(dot(g0,gpred)/size1)
            gpred -= (lr_g * delta / vv) * v
            approx(dfgold, dot(gpred,v)) || error("dfgold2=$dfgold dfpred2=$(dot(gpred,v))")
            size2 = vecnorm(gpred)
            isapprox(size2, size0, rtol=.1, atol=1e-5) || error("size0=$size0 size2=$size2")

            # adjust curvature using newton step
            v = (-1/hpred)*gpred
            vv = sumabs2(v)
            w3 = w+v
            f3 = f(w3,x,y)
            dfgold = f3-f0
            dfpred = dot(gpred,v) + 0.5 * hpred * vv
            dfpred < 0 || error("dfpred3=$dfpred dot=$(dot(gpred,v)) hpred=$hpred vv=$vv")
            delta = (dfpred - dfgold) # =0.5*(hpred-hgold)*vv
            hpred1 = hpred
            hpred2 = hpred - (2 * delta / vv)
            approx(dfgold, dot(gpred,v)+0.5*hpred2*vv) || error("dfgold3=$dfgold dfpred3=$(dot(gpred,v)+0.5*hpred2*vv)")
            hpred3 = hpred - (lr_h * 2 * delta / vv)
            if hpred2 >= 0 # 0.001
                hpred = hpred2 # clamp(hpred3, 0.01, 10.0)
            elseif hpred2 < 0
                warn("hpred=$hpred hpred2=$hpred2 delta=$delta vv=$vv dfpred=$dfpred dfgold=$dfgold")
            end

            # finally take that step
            v = (-lr/hpred)*gpred
            w4 = w+v
            f4 = f(w4,x,y)
            if f4 <= f0
                w = w4
                gpred += hpred * v
            else
                warn("f0=$f0 f4=$f4 hpred=$hpred gpred=$(vecnorm(gpred))")
            end

            if epoch in pr
                @printf("f=%g->%g vcos=%g->%g size=%g->%g curv=%g->%g delta=%g\n",
                        f0,f4,vcos1,vcos2,size1,size2,hpred1,hpred2,delta)
            end

            # gpred -= (lr_g * delta / (2*vv)) * v
            # hpred -= (lr_h * delta / vv)

            # if 0.01 < hpred2 < 100.0    # do not let hpred get too small or big
            #     hpred = hpred2
            # end
            # if f2 <= f0
            #     w = w2
            #     gpred += (hpred - lr_g * delta / (2*vv)) * v
            #     # gpred -= (lr_g * delta / (2*vv)) * v
            #     # gpred += hpred * v    # we assume hpred stays constant, but gpred should be updated after the step
            # else
            #     fill!(gpred,0)
            #     delta < 0 || error("delta=$delta")
            #     # (0.01<hpred<100) && (hpred -= (lr_h * delta / vv))
            #     # hpred = hpred0
            #     # gpred *= -1
            #     # lr *= 0.99f0
            #     # lr < (1e-6) && (lr=lr0; fill!(gpred,0))
            # end
        end
        report(epoch)
        loss1 = mean(f(w,x,y) for (x,y) in dtst)
        loss1 < loss0 || (lr *= decay)
        loss0 = loss1
    end
    return w
end


# SGD that takes a newton step in gradient direction every iteration:
# gets stuck around 0.39 with lr=1.0.
# gets stuck around 0.31 with lr=0.5.
# gets stuck around 0.2775 with lr=0.1.
# reaches 0.2721 with lr=0.1, decay=0.5.
# vcos usually negative.

function supersgd(w=nothing; batch=100, epochs=100, atype=KnetArray{Float64}, hpred=0.4, pr=0, lr=1.0, decay=1.0)
    Knet.rng(true)
    dtrn = minibatch(xtrn,ytrn,batch;atype=atype)
    dtst = minibatch(xtst,ytst,length(ytst);atype=atype)
    f(w,x,y) = -sum(y .* logp(w*x,1))/size(y,2); gradf = grad(f)
    w==nothing && (w = convert(atype, zeros(10,784)))
    gpred = zeros(w); loss0=loss1=Inf
    report(epoch)=println(map((epoch,
                               :lr, lr,
                               :loss, mean(f(w,x,y) for (x,y) in dtst),
                               :vcos, mean(vcos(gpred,gradf(w,x,y)) for (x,y) in dtst),
                               :ggold, mean(vecnorm(gradf(w,x,y)) for (x,y) in dtst),
                               :gpred, vecnorm(gpred), :hpred, hpred
                               )) do x; isa(x,AbstractFloat) ? parse(@sprintf("%g",x)) : x; end)
    report(0)
    approx(a,b)=isapprox(a,b;rtol=1e-3,atol=1e-12)
    for epoch=1:epochs
        for (x,y) in dtrn
            f0 = f(w,x,y)
            gpred = gradf(w,x,y)   # perfect direction and size, only need curvature

            # adjust curvature using newton step
            v = (-1/hpred)*gpred
            vv = sumabs2(v)
            w3 = w+v
            f3 = f(w3,x,y)
            dfgold = f3-f0
            dfpred = dot(gpred,v) + 0.5 * hpred * vv
            dfpred < 0 || error("dfpred3=$dfpred dot=$(dot(gpred,v)) hpred=$hpred vv=$vv")
            delta = (dfpred - dfgold) # =0.5*(hpred-hgold)*vv
            hpred1 = hpred
            hpred2 = hpred - (2 * delta / vv)
            approx(dfgold, dot(gpred,v)+0.5*hpred2*vv) || error("dfgold3=$dfgold dfpred3=$(dot(gpred,v)+0.5*hpred2*vv)")
            if hpred2 >= 0 # 0.001
                hpred = hpred2 # clamp(hpred3, 0.01, 10.0)
            elseif hpred2 < 0
                warn("hpred=$hpred hpred2=$hpred2 delta=$delta vv=$vv dfpred=$dfpred dfgold=$dfgold")
            end

            # finally take that step
            v = (-lr/hpred)*gpred
            w4 = w+v
            f4 = f(w4,x,y)
            if f4 <= f0
                w = w4
                # gpred += hpred * v
            else
                warn("f0=$f0 f4=$f4 hpred=$hpred gpred=$(vecnorm(gpred))")
            end

            if epoch in pr
                @printf("f=%g->%g vcos=%g->%g size=%g->%g curv=%g->%g delta=%g\n",
                        f0,f4,vcos1,vcos2,size1,size2,hpred1,hpred2,delta)
            end

        end
        report(epoch)
        loss1 = mean(f(w,x,y) for (x,y) in dtst)
        loss1 < loss0 || (lr *= decay)
        loss0 = loss1
    end
    return w
end

# gets to loss=0.2730 in 100 epochs with batch=100 lr=0.1
# gets to loss=0.2715 with lr=0.1, decay=0.5.
# vcos between test gradient and training minibatches is around 0.1.
# vcos changes when lr changes, looks like a bug!

function sgd(; batch=100, lr=0.1, epochs=100, atype=KnetArray{Float32}, decay=0.5)
    dtrn = minibatch(xtrn,ytrn,batch;atype=atype)
    dtst = minibatch(xtst,ytst,length(ytst);atype=atype)
    f(w,x,y) = -sum(y .* logp(w*x,1))/size(y,2)
    g = grad(f)
    w = convert(atype, zeros(10,784))
    println((0, mean([f(w,x,y) for (x,y) in dtst])))
    loss1 = loss0 = Inf
    for epoch=1:epochs
        vcos1 = Any[]
        for (x,y) in dtrn
            dw = g(w,x,y)
            push!(vcos1, mean(vcos(dw,g(w,x,y)) for (x,y) in dtst))
            axpy!(-lr,dw,w)
        end
        loss1 = mean([f(w,x,y) for (x,y) in dtst])
        println((epoch, :lr, lr, :loss, loss1, :vcos, mean(vcos1)))
        loss1 > loss0 && (lr *= decay)
        loss0 = loss1
    end
    return w
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
